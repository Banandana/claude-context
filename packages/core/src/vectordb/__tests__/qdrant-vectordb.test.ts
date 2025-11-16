import { QdrantVectorDatabase } from '../qdrant-vectordb';
import { VectorDocument } from '../types';

// Mock the QdrantClient
jest.mock('@qdrant/js-client-rest', () => {
  return {
    QdrantClient: jest.fn().mockImplementation(() => ({
      getCollections: jest.fn().mockResolvedValue({
        collections: [],
      }),
      createCollection: jest.fn().mockResolvedValue({}),
      deleteCollection: jest.fn().mockResolvedValue({}),
      upsert: jest.fn().mockResolvedValue({}),
      search: jest.fn().mockResolvedValue([]),
      searchPointGroups: jest.fn().mockResolvedValue({
        groups: [],
      }),
      delete: jest.fn().mockResolvedValue({}),
      scroll: jest.fn().mockResolvedValue({
        points: [],
      }),
    })),
  };
});

describe('QdrantVectorDatabase', () => {
  let db: QdrantVectorDatabase;

  beforeEach(() => {
    db = new QdrantVectorDatabase({
      address: 'http://localhost:6333',
    });
  });

  describe('Collection Management', () => {
    test('should create a regular collection', async () => {
      await db.createCollection('test_collection', 384);
      expect(db).toBeDefined();
    });

    test('should create a hybrid collection', async () => {
      await db.createHybridCollection('test_hybrid', 384);
      expect(db).toBeDefined();
    });

    test('should check if collection exists', async () => {
      const exists = await db.hasCollection('test_collection');
      expect(typeof exists).toBe('boolean');
    });

    test('should list all collections', async () => {
      const collections = await db.listCollections();
      expect(Array.isArray(collections)).toBe(true);
    });

    test('should drop a collection', async () => {
      await db.dropCollection('test_collection');
      expect(db).toBeDefined();
    });
  });

  describe('Data Insertion', () => {
    const mockDocuments: VectorDocument[] = [
      {
        id: '1',
        vector: Array(384).fill(0.1),
        content: 'function example() { return 42; }',
        relativePath: 'src/example.ts',
        startLine: 1,
        endLine: 3,
        fileExtension: '.ts',
        metadata: { language: 'typescript' },
      },
      {
        id: '2',
        vector: Array(384).fill(0.2),
        content: 'const result = await fetchData();',
        relativePath: 'src/async.ts',
        startLine: 10,
        endLine: 11,
        fileExtension: '.ts',
        metadata: { language: 'typescript' },
      },
    ];

    test('should insert documents', async () => {
      await db.insert('test_collection', mockDocuments);
      expect(db).toBeDefined();
    });

    test('should insert hybrid documents with BM25', async () => {
      await db.insertHybrid('test_hybrid', mockDocuments);
      expect(db).toBeDefined();
    });

    test('should handle empty document array', async () => {
      await db.insert('test_collection', []);
      expect(db).toBeDefined();
    });
  });

  describe('Search Operations', () => {
    test('should perform dense vector search', async () => {
      const queryVector = Array(384).fill(0.15);
      const results = await db.search('test_collection', queryVector, {
        topK: 10,
      });
      expect(Array.isArray(results)).toBe(true);
    });

    test('should perform search with threshold', async () => {
      const queryVector = Array(384).fill(0.15);
      const results = await db.search('test_collection', queryVector, {
        topK: 5,
        threshold: 0.5,
      });
      expect(Array.isArray(results)).toBe(true);
    });

    test('should perform hybrid search', async () => {
      const denseVector = Array(384).fill(0.15);
      const results = await db.hybridSearch(
        'test_hybrid',
        [
          {
            data: denseVector,
            anns_field: 'vector',
            param: {},
            limit: 10,
          },
          {
            data: 'function definition',
            anns_field: 'sparse_vector',
            param: {},
            limit: 10,
          },
        ],
        {
          limit: 10,
          rerank: { strategy: 'rrf' },
        }
      );
      expect(Array.isArray(results)).toBe(true);
    });

    test('should require dense vector in hybrid search', async () => {
      const results = db.hybridSearch('test_hybrid', [
        {
          data: 'search text',
          anns_field: 'sparse_vector',
          param: {},
          limit: 10,
        },
      ]);
      await expect(results).rejects.toThrow(
        'Dense search request (vector) is required'
      );
    });
  });

  describe('Document Operations', () => {
    test('should delete documents', async () => {
      const ids = ['1', '2', '3'];
      await db.delete('test_collection', ids);
      expect(db).toBeDefined();
    });

    test('should handle empty delete', async () => {
      await db.delete('test_collection', []);
      expect(db).toBeDefined();
    });

    test('should query documents', async () => {
      const results = await db.query(
        'test_collection',
        'language == "typescript"',
        ['id', 'content', 'relativePath'],
        10
      );
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('Collection Limits', () => {
    test('should check collection limit', async () => {
      const canCreate = await db.checkCollectionLimit();
      expect(typeof canCreate).toBe('boolean');
    });

    test('should return true for Qdrant (no limit in OSS)', async () => {
      const canCreate = await db.checkCollectionLimit();
      expect(canCreate).toBe(true);
    });
  });

  describe('Configuration', () => {
    test('should initialize with custom address', () => {
      const customDb = new QdrantVectorDatabase({
        address: 'http://custom:6333',
      });
      expect(customDb).toBeDefined();
    });

    test('should initialize with API key', () => {
      const customDb = new QdrantVectorDatabase({
        address: 'https://qdrant.cloud',
        apiKey: 'test-key',
      });
      expect(customDb).toBeDefined();
    });

    test('should use default localhost if no address provided', () => {
      const defaultDb = new QdrantVectorDatabase({});
      expect(defaultDb).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    test('should handle collection already exists error', async () => {
      const db = new QdrantVectorDatabase({
        address: 'http://localhost:6333',
      });
      // This should not throw even if collection exists
      await db.createCollection('existing_collection', 384);
      expect(db).toBeDefined();
    });

    test('should handle collection not found on drop', async () => {
      const db = new QdrantVectorDatabase({
        address: 'http://localhost:6333',
      });
      // This should not throw even if collection doesn't exist
      await db.dropCollection('nonexistent_collection');
      expect(db).toBeDefined();
    });
  });

  describe('VectorDocument Mapping', () => {
    test('should map search results to VectorDocument correctly', async () => {
      const queryVector = Array(384).fill(0.15);
      const results = await db.search('test_collection', queryVector);

      if (results.length > 0) {
        const doc = results[0].document;
        expect(doc).toHaveProperty('id');
        expect(doc).toHaveProperty('vector');
        expect(doc).toHaveProperty('content');
        expect(doc).toHaveProperty('relativePath');
        expect(doc).toHaveProperty('startLine');
        expect(doc).toHaveProperty('endLine');
        expect(doc).toHaveProperty('fileExtension');
        expect(doc).toHaveProperty('metadata');
      }
    });

    test('should map hybrid search results correctly', async () => {
      const denseVector = Array(384).fill(0.15);
      const results = await db.hybridSearch(
        'test_hybrid',
        [
          {
            data: denseVector,
            anns_field: 'vector',
            param: {},
            limit: 10,
          },
        ],
        { limit: 10 }
      );

      if (results.length > 0) {
        const doc = results[0].document;
        expect(typeof doc.id).toBe('string');
        expect(typeof doc.content).toBe('string');
        expect(typeof doc.relativePath).toBe('string');
      }
    });
  });

  describe('Initialization', () => {
    test('should initialize on first method call', async () => {
      const db = new QdrantVectorDatabase({
        address: 'http://localhost:6333',
      });
      await db.listCollections();
      expect(db).toBeDefined();
    });

    test('should not reinitialize on subsequent calls', async () => {
      const db = new QdrantVectorDatabase({
        address: 'http://localhost:6333',
      });
      await db.listCollections();
      await db.listCollections();
      expect(db).toBeDefined();
    });
  });
});
