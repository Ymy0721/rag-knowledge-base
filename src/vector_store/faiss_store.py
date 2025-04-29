import os
import json
import sqlite3
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import faiss
import gc
from src.utils import logger
import pandas as pd # <--- 新增: 导入 pandas 以检查 Timestamp 类型

class FAISSVectorStore:
    """使用FAISS实现向量存储，支持SQLite存储文档和按需加载"""

    def __init__(self, config: Dict):
        """
        初始化FAISS向量存储
        """
        self.config = config
        self.index: Optional[faiss.Index] = None
        self.dimension = config['embedding']['dimension']
        self.index_type = config['faiss']['index_type']
        self.vector_store_path = os.path.join(os.getcwd(), config['data']['vector_store_path'])
        self.index_path = os.path.join(self.vector_store_path, "faiss.index")
        self.documents_path = os.path.join(self.vector_store_path, config['data']['documents_filename']) # <--- 修改: 指向 .db 文件

        self.lazy_loading = config.get('memory', {}).get('lazy_loading', False)
        self.unload_after_query = config.get('memory', {}).get('unload_after_query', False)
        self.use_mmap = config.get('faiss', {}).get('use_mmap', False) # 仍然读取配置，但load_index会忽略它

        self._index_loaded = False
        self._documents_loaded = False # 现在表示 SQLite 文件是否已验证存在
        self._initialized = False

        # 数据库连接（通常在需要时创建）
        self._db_conn: Optional[sqlite3.Connection] = None

    def _connect_db(self) -> sqlite3.Connection:
        """建立到SQLite数据库的连接"""
        try:
            conn = sqlite3.connect(self.documents_path)
            # 设置 row_factory 以便按列名访问
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite DB at {self.documents_path}: {e}")
            raise

    def _close_db(self, conn: Optional[sqlite3.Connection]):
        """关闭SQLite数据库连接"""
        if conn:
            try:
                conn.close()
            except sqlite3.Error as e:
                logger.error(f"Error closing SQLite DB connection: {e}")

    def create_index(self):
        """创建新的FAISS索引，支持IndexIVFPQ"""
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = self.config['faiss']['nlist']
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        elif self.index_type == "IndexIVFPQ": # <--- 新增: 处理 IndexIVFPQ
            quantizer = faiss.IndexFlatL2(self.dimension) # 或 IndexFlatIP，取决于你的相似度度量
            nlist = self.config['faiss']['nlist']
            m = self.config['faiss']['pq_m']
            nbits = self.config['faiss']['pq_nbits']
            metric = faiss.METRIC_L2 # 或 faiss.METRIC_INNER_PRODUCT
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, metric)
            logger.info(f"Using IVF{nlist},PQ{m}x{nbits} index.")
        else:
            logger.warning(f"Unsupported index_type '{self.index_type}'. Defaulting to IndexFlatL2.")
            self.index = faiss.IndexFlatL2(self.dimension)

        logger.info(f"Created FAISS index of type {self.index_type} with dimension {self.dimension}")
        return self.index

    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        将文档向量添加到FAISS索引，并将文档内容存储到SQLite数据库。
        处理元数据中的Timestamp对象。
        """
        if self.index is None:
            self.create_index()

        # 确保向量是float32类型
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # 对于IP索引类型，归一化向量 (包括IVFPQ使用IP度量的情况)
        metric_type = getattr(self.index, 'metric_type', None)
        if metric_type == faiss.METRIC_INNER_PRODUCT:
             logger.info("Normalizing embeddings for METRIC_INNER_PRODUCT index.")
             faiss.normalize_L2(embeddings)

        # 对于IVF索引，需要先训练
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            train_size = self.config['faiss'].get('train_size', embeddings.shape[0])
            train_vectors = embeddings[:train_size]
            if train_vectors.shape[0] < self.config['faiss']['nlist']:
                 logger.warning(f"Number of training vectors ({train_vectors.shape[0]}) is less than nlist ({self.config['faiss']['nlist']}). Training might be suboptimal.")
            # 确保训练向量也归一化（如果需要）
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                 faiss.normalize_L2(train_vectors)

            logger.info(f"Training {self.index_type} index with {train_vectors.shape[0]} vectors...")
            self.index.train(train_vectors)
            logger.info("Index training complete.")

        # --- SQLite 操作 ---
        conn = None
        try:
            conn = self._connect_db()
            cursor = conn.cursor()
            # 创建表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    faiss_id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.commit()

            # 逐个添加向量和文档记录
            num_added = 0
            start_faiss_id = self.index.ntotal # 获取当前索引中的向量数作为起始ID
            logger.info(f"Starting FAISS ID: {start_faiss_id}")

            insert_sql = "INSERT INTO documents (faiss_id, text, metadata) VALUES (?, ?, ?)"
            doc_data_to_insert = []

            # --- 新增: JSON 序列化辅助函数 ---
            def json_serializer(obj):
                """JSON serializer for objects not serializable by default json code"""
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat() # 将 Timestamp 转换为 ISO 格式字符串
                raise TypeError (f"Object of type {obj.__class__.__name__} is not JSON serializable")
            # --- 结束: JSON 序列化辅助函数 ---

            for i, doc in enumerate(documents):
                current_faiss_id = start_faiss_id + i
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                # --- 修改: 使用自定义序列化器 ---
                try:
                    metadata_str = json.dumps(metadata, default=json_serializer) # 存储元数据为JSON字符串
                except TypeError as e:
                     logger.error(f"Error serializing metadata for document index {i} (FAISS ID {current_faiss_id}): {e}. Metadata: {metadata}")
                     metadata_str = json.dumps({"error": "Metadata serialization failed"}) # 存入错误信息
                # --- 结束: 修改 ---
                doc_data_to_insert.append((current_faiss_id, text, metadata_str))

            # 批量插入数据库
            if doc_data_to_insert:
                 cursor.executemany(insert_sql, doc_data_to_insert)
                 conn.commit()
                 logger.info(f"Inserted {len(doc_data_to_insert)} documents into SQLite.")

            # 添加向量到FAISS索引
            self.index.add(embeddings)
            num_added = len(embeddings)
            logger.info(f"Added {num_added} vectors to FAISS index. New total: {self.index.ntotal}")

        except sqlite3.Error as e:
            logger.error(f"SQLite error during add_documents: {e}")
            if conn:
                conn.rollback() # 回滚事务
            raise # 重新抛出异常，让调用者知道出错了
        finally:
            self._close_db(conn)
        # --- 结束 SQLite 操作 ---

        logger.info(f"Finished adding batch. FAISS index now contains {self.index.ntotal} vectors.")


    def initialize(self):
        """
        初始化向量存储，验证索引和SQLite数据库文件是否存在。
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.documents_path):
            raise FileNotFoundError(f"Index file ({self.index_path}) or documents DB ({self.documents_path}) not found.")

        logger.info(f"Vector store initialized. Index file and documents DB exist.")
        self._initialized = True
        return True

    def load_index(self) -> bool:
        """
        只加载FAISS索引。现在忽略 use_mmap 配置。
        """
        if self._index_loaded and self.index is not None:
            return True

        if not os.path.exists(self.index_path):
            logger.error(f"FAISS index file not found: {self.index_path}")
            return False

        try:
            logger.info(f"Loading FAISS index from {self.index_path} (mmap disabled)")
            self.index = faiss.read_index(self.index_path) # <--- 修改: 移除 mmap flag

            # 为IVF索引设置nprobe
            if isinstance(self.index, faiss.IndexIVF):
                self.index.nprobe = self.config['faiss'].get('nprobe', 10)
                logger.info(f"Set nprobe={self.index.nprobe} for IVF index.")

            self._index_loaded = True
            logger.info(f"FAISS index loaded successfully. Type: {type(self.index)}, Trained: {self.index.is_trained}, Total vectors: {self.index.ntotal}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            self.index = None
            self._index_loaded = False
            return False

    def load_documents(self) -> bool:
        """
        加载文档数据。
        如果启用了惰性加载，则仅检查SQLite文件是否存在。
        否则（非惰性加载），也只检查文件是否存在（按需连接）。
        """
        if self._documents_loaded:
             return True

        if not os.path.exists(self.documents_path):
            logger.error(f"Documents SQLite DB file not found: {self.documents_path}")
            return False

        logger.info(f"Documents SQLite DB file exists at {self.documents_path}. Ready for on-demand access.")
        self._documents_loaded = True # 标记为“已加载”（逻辑上）
        return True

    def load(self):
        """从磁盘加载索引并验证文档数据库"""
        if not self._index_loaded:
            if not self.load_index():
                 raise RuntimeError(f"Failed to load FAISS index from {self.index_path}")

        if not self._documents_loaded:
             if not self.load_documents():
                 raise RuntimeError(f"Documents SQLite DB file check failed at {self.documents_path}.")

        logger.info(f"Vector store load sequence complete. Index loaded: {self._index_loaded}, Documents DB verified: {self._documents_loaded}")
        return True

    def unload_index(self):
        """卸载索引以释放内存"""
        if self.index is not None:
            index_type = type(self.index)
            del self.index
            self.index = None
            self._index_loaded = False
            gc.collect()
            logger.info(f"FAISS index ({index_type}) unloaded from memory")

    def unload_documents(self):
        """
        卸载文档数据相关的资源。
        对于SQLite，主要是关闭可能存在的持久连接（如果使用了的话）。
        在按需连接模式下，此方法可能只需重置状态。
        """
        if self._documents_loaded:
            self._documents_loaded = False
            logger.info("Documents DB status reset (connections are on-demand).")
        gc.collect()

    def unload(self):
        """卸载索引和文档资源"""
        self.unload_index()
        self.unload_documents()
        logger.info("Vector store completely unloaded from memory")

    def _get_documents_from_storage(self, indices: List[int]) -> List[Optional[Dict]]:
        """
        根据FAISS ID列表从SQLite数据库中检索文档。
        返回与输入indices顺序一致的列表，如果某个ID未找到，则对应位置为None。
        """
        if not indices:
            return []

        results_map: Dict[int, Dict] = {}
        conn = None
        try:
            conn = self._connect_db()
            cursor = conn.cursor()

            placeholders = ','.join('?' * len(indices))
            query = f"SELECT faiss_id, text, metadata FROM documents WHERE faiss_id IN ({placeholders})"

            cursor.execute(query, indices)
            rows = cursor.fetchall()

            for row in rows:
                faiss_id = row['faiss_id']
                metadata = {}
                try:
                    metadata_str = row['metadata']
                    if metadata_str:
                        metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata JSON for faiss_id {faiss_id}")
                except Exception as e:
                     logger.warning(f"Error processing metadata for faiss_id {faiss_id}: {e}")

                results_map[faiss_id] = {
                    "text": row['text'],
                    "metadata": metadata
                }

        except sqlite3.Error as e:
            logger.error(f"SQLite error during document retrieval: {e}")
            results_map = {}
        finally:
            self._close_db(conn)

        final_results = []
        for idx in indices:
            final_results.append(results_map.get(idx))

        return final_results

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """搜索最相似的向量，从SQLite检索文档，并确保结果按相似度排序"""
        if not self._index_loaded:
            logger.info("Index not loaded. Attempting to load index...")
            if not self.load_index():
                 raise ValueError("Failed to load index for searching.")
        if self.index is None:
            raise ValueError("Index is None even after load attempt. Cannot search.")

        if not self._documents_loaded:
             logger.info("Documents DB not verified. Attempting to verify...")
             if not self.load_documents():
                  raise ValueError("Failed to verify documents DB for searching.")

        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        metric_type = getattr(self.index, 'metric_type', None)
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(query_vector)

        logger.debug(f"Searching index with k={k}")
        distances, indices = self.index.search(query_vector, k)
        logger.debug(f"Search completed. Found indices: {indices[0]}")

        matching_docs_data = []
        valid_indices = [int(idx) for idx in indices[0] if idx != -1]

        if valid_indices:
             retrieved_docs_or_nones = self._get_documents_from_storage(valid_indices)

             retrieval_map = {idx: doc for idx, doc in zip(valid_indices, retrieved_docs_or_nones)}

             for i, idx_float in enumerate(indices[0]):
                 idx = int(idx_float)
                 if idx != -1:
                     retrieved_doc = retrieval_map.get(idx)
                     if retrieved_doc is not None:
                         doc = retrieved_doc.copy()
                         doc["score"] = float(distances[0][i])
                         matching_docs_data.append(doc)
                     else:
                         logger.warning(f"Index {idx} returned by FAISS search (score: {distances[0][i]:.4f}), but document could not be retrieved from storage (SQLite).")

        is_ip_metric = (metric_type == faiss.METRIC_INNER_PRODUCT)

        if is_ip_metric:
            matching_docs_data.sort(key=lambda x: x['score'], reverse=True)
            logger.debug("Sorted results by score (descending) for IP metric.")
        else:
            matching_docs_data.sort(key=lambda x: x['score'], reverse=False)
            logger.debug("Sorted results by score (ascending) for L2/other metric.")

        if self.unload_after_query:
            logger.info("Unloading document resources after query (if any were held).")
            self.unload_documents()

        return distances[0], matching_docs_data

    def save(self):
        """保存FAISS索引到磁盘。文档已在add_documents期间存入SQLite."""
        if self.index is None:
            logger.error("Index is None. Cannot save.")
            return

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        logger.info(f"Saving FAISS index to {self.index_path}")
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
             logger.error(f"Error saving FAISS index: {e}")
             raise
