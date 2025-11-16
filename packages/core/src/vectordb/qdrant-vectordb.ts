import { QdrantClient } from '@qdrant/js-client-rest';
import {
  VectorDatabase,
  VectorDocument,
  SearchOptions,
  VectorSearchResult,
  HybridSearchRequest,
  HybridSearchOptions,
  HybridSearchResult,
} from './types';

export interface QdrantConfig {
  address?: string;
  apiKey?: string;
}

export class QdrantVectorDatabase implements VectorDatabase {
  private client: QdrantClient;
  private initialized = false;

  constructor(config: QdrantConfig = {}) {
    const url = config.address || process.env.QDRANT_URL || 'http://localhost:6333';
    this.client = new QdrantClient({
      url,
      apiKey: config.apiKey,
    });
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    try {
      await this.client.getCollections();
      this.initialized = true;
    } catch (error) {
      throw new Error(`Failed to initialize Qdrant client: ${error}`);
    }
  }

  async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
    await this.initialize();
    try {
      await this.client.createCollection(collectionName, {
        vectors: {
          size: dimension,
          distance: 'Cosine',
        },
      });
    } catch (error: any) {
      if (error.message?.includes('already exists')) {
        return;
      }
      throw new Error(`Failed to create collection ${collectionName}: ${error.message}`);
    }
  }

  async createHybridCollection(
    collectionName: string,
    dimension: number,
    description?: string
  ): Promise<void> {
    await this.initialize();
    try {
      await this.client.createCollection(collectionName, {
        vectors: {
          size: dimension,
          distance: 'Cosine',
        },
        sparse_vectors: {
          text: {},
        },
      });
    } catch (error: any) {
      if (error.message?.includes('already exists')) {
        return;
      }
      throw new Error(`Failed to create hybrid collection ${collectionName}: ${error.message}`);
    }
  }

  async dropCollection(collectionName: string): Promise<void> {
    await this.initialize();
    try {
      await this.client.deleteCollection(collectionName);
    } catch (error: any) {
      if (error.message?.includes('not found')) {
        return;
      }
      throw new Error(`Failed to drop collection ${collectionName}: ${error.message}`);
    }
  }

  async hasCollection(collectionName: string): Promise<boolean> {
    await this.initialize();
    try {
      const collections = await this.client.getCollections();
      return collections.collections.some((c) => c.name === collectionName);
    } catch (error) {
      throw new Error(`Failed to check collection existence: ${error}`);
    }
  }

  async listCollections(): Promise<string[]> {
    await this.initialize();
    try {
      const collections = await this.client.getCollections();
      return collections.collections.map((c) => c.name);
    } catch (error) {
      throw new Error(`Failed to list collections: ${error}`);
    }
  }

  async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
    await this.initialize();
    if (documents.length === 0) return;

    try {
      const points = documents.map((doc) => ({
        id: this.stringToUUID(doc.id),
        vector: doc.vector,
        payload: {
          originalId: doc.id, // Store original ID in payload for reference
          content: doc.content,
          relativePath: doc.relativePath,
          startLine: doc.startLine,
          endLine: doc.endLine,
          fileExtension: doc.fileExtension,
          metadata: doc.metadata,
        },
      }));

      await this.client.upsert(collectionName, {
        points,
      });
    } catch (error) {
      throw new Error(`Failed to insert documents into ${collectionName}: ${error}`);
    }
  }

  async insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void> {
    await this.initialize();
    if (documents.length === 0) return;

    try {
      const points = documents.map((doc) => ({
        id: this.stringToUUID(doc.id),
        vector: doc.vector, // Unnamed default dense vector
        sparse_vector: { // Note: singular form for Qdrant API compatibility
          text: {
            indices: this.tokenizeToIndices(doc.content),
            values: this.tokenizeToValues(doc.content),
          },
        },
        payload: {
          originalId: doc.id, // Store original ID in payload for reference
          content: doc.content,
          relativePath: doc.relativePath,
          startLine: doc.startLine,
          endLine: doc.endLine,
          fileExtension: doc.fileExtension,
          metadata: doc.metadata,
        },
      }));

      await this.client.upsert(collectionName, {
        points,
      });
    } catch (error) {
      throw new Error(`Failed to insert hybrid documents into ${collectionName}: ${error}`);
    }
  }

  async search(
    collectionName: string,
    queryVector: number[],
    options?: SearchOptions
  ): Promise<VectorSearchResult[]> {
    await this.initialize();
    try {
      const limit = options?.topK || 10;
      const scoreThreshold = options?.threshold;

      const response = await this.client.search(collectionName, {
        vector: queryVector,
        limit,
        score_threshold: scoreThreshold,
        with_payload: true,
      });

      return response.map((point: any) => ({
        document: {
          id: (point.payload?.originalId || point.id) as string, // Use original ID from payload
          vector: queryVector,
          content: (point.payload?.content || '') as string,
          relativePath: (point.payload?.relativePath || '') as string,
          startLine: (point.payload?.startLine || 0) as number,
          endLine: (point.payload?.endLine || 0) as number,
          fileExtension: (point.payload?.fileExtension || '') as string,
          metadata: (point.payload?.metadata || {}) as Record<string, any>,
        },
        score: point.score || 0,
      }));
    } catch (error) {
      throw new Error(`Failed to search in ${collectionName}: ${error}`);
    }
  }

  async hybridSearch(
    collectionName: string,
    searchRequests: HybridSearchRequest[],
    options?: HybridSearchOptions
  ): Promise<HybridSearchResult[]> {
    await this.initialize();

    try {
      const limit = options?.limit || 10;
      const denseRequest = searchRequests.find((r) => r.anns_field === 'vector');
      const sparseRequest = searchRequests.find((r) => r.anns_field === 'sparse_vector');

      if (!denseRequest) {
        throw new Error('Dense search request (vector) is required for hybrid search');
      }

      const denseVector = denseRequest.data as number[];
      const sparseText = sparseRequest?.data as string | undefined;

      // Perform dense vector search
      const denseResults = await this.client.search(collectionName, {
        vector: denseVector,
        limit: limit * 2, // Get more results for better fusion
        with_payload: true,
      });

      let allResults: Map<string, { point: any; denseRank?: number; sparseRank?: number }> = new Map();

      // Add dense results with rankings
      denseResults.forEach((result, index) => {
        const id = result.id as string;
        allResults.set(id, {
          point: result,
          denseRank: index + 1,
        });
      });

      // Perform sparse vector search if sparse text is provided
      if (sparseText) {
        const sparseVector = this.tokenizeToVector(sparseText);
        
        // Search using sparse vector
        const sparseResults = await this.client.search(collectionName, {
          vector: {
            name: 'text',
            vector: sparseVector,
          },
          limit: limit * 2,
          with_payload: true,
        });

        // Add sparse results with rankings
        sparseResults.forEach((result, index) => {
          const id = result.id as string;
          const existing = allResults.get(id);
          if (existing) {
            existing.sparseRank = index + 1;
          } else {
            allResults.set(id, {
              point: result,
              sparseRank: index + 1,
            });
          }
        });
      }

      // Apply RRF (Reciprocal Rank Fusion) scoring
      const k = options?.rerank?.params?.k || 60;
      const scoredResults = Array.from(allResults.values()).map((item) => {
        let rrfScore = 0;
        
        // Add dense contribution
        if (item.denseRank !== undefined) {
          rrfScore += 1 / (k + item.denseRank);
        }
        
        // Add sparse contribution  
        if (item.sparseRank !== undefined) {
          rrfScore += 1 / (k + item.sparseRank);
        }

        return {
          point: item.point,
          rrfScore,
        };
      });

      // Sort by RRF score and take top results
      scoredResults.sort((a, b) => b.rrfScore - a.rrfScore);
      const topResults = scoredResults.slice(0, limit);

      // Convert to HybridSearchResult format
      return topResults.map((item) => ({
        document: {
          id: (item.point.payload?.originalId || item.point.id) as string,
          vector: denseVector,
          content: item.point.payload?.content || '',
          relativePath: item.point.payload?.relativePath || '',
          startLine: item.point.payload?.startLine || 0,
          endLine: item.point.payload?.endLine || 0,
          fileExtension: item.point.payload?.fileExtension || '',
          metadata: item.point.payload?.metadata || {},
        },
        score: item.rrfScore, // Use RRF score
      }));
    } catch (error) {
      throw new Error(`Failed to perform hybrid search in ${collectionName}: ${error}`);
    }
  }

  async delete(collectionName: string, ids: string[]): Promise<void> {
    await this.initialize();
    if (ids.length === 0) return;

    try {
      // Convert string IDs to UUIDs
      const pointIds = ids.map((id) => this.stringToUUID(id));

      await this.client.delete(collectionName, {
        points: pointIds,
      });
    } catch (error) {
      throw new Error(`Failed to delete documents from ${collectionName}: ${error}`);
    }
  }

  async query(
    collectionName: string,
    filter: string,
    outputFields: string[],
    limit?: number
  ): Promise<Record<string, any>[]> {
    await this.initialize();
    try {
      // For now, return all points with payloads since Qdrant doesn't have a simple query API
      // A production implementation would parse the filter expression
      const response = await this.client.scroll(collectionName, {
        limit: limit || 100,
        with_payload: true,
      });

      return response.points.map((point) => ({
        id: point.id,
        ...point.payload,
      }));
    } catch (error) {
      throw new Error(`Failed to query ${collectionName}: ${error}`);
    }
  }

  async checkCollectionLimit(): Promise<boolean> {
    await this.initialize();
    try {
      const collections = await this.client.getCollections();
      // Qdrant has no collection limit in open source version
      // Return true to indicate we can create more collections
      return true;
    } catch (error) {
      throw new Error(`Failed to check collection limit: ${error}`);
    }
  }

  // Helper method to tokenize text into sparse vector indices
  private tokenizeToIndices(text: string): number[] {
    const tokens = this.tokenize(text);
    const indices: number[] = [];
    const seen = new Set<number>();

    tokens.forEach((token) => {
      const index = this.hashToken(token);
      if (!seen.has(index)) {
        indices.push(index);
        seen.add(index);
      }
    });

    return indices;
  }

  // Helper method to tokenize text into sparse vector values
  private tokenizeToValues(text: string): number[] {
    const tokens = this.tokenize(text);
    const indices = this.tokenizeToIndices(text); // Get indices in correct order
    const tokenMap = new Map<number, number>();

    tokens.forEach((token) => {
      const index = this.hashToken(token);
      tokenMap.set(index, (tokenMap.get(index) || 0) + 1);
    });

    // Return values in the same order as indices
    return indices.map(index => tokenMap.get(index) || 0);
  }

  // Helper method to tokenize text for BM25
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter((token) => token.length > 0);
  }

  // Hash token to get consistent index
  private hashToken(token: string): number {
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      const char = token.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash) % 65536; // Keep index within reasonable range
  }

  // Helper method to convert string ID to UUID format for Qdrant
  // Qdrant requires point IDs to be either UUIDs or unsigned integers
  private stringToUUID(id: string): string {
    const crypto = require('crypto');
    // Create a consistent hash from the ID
    const hash = crypto.createHash('md5').update(id).digest('hex');
    // Format as UUID v4: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    return `${hash.substring(0, 8)}-${hash.substring(8, 12)}-4${hash.substring(13, 16)}-${hash.substring(16, 20)}-${hash.substring(20, 32)}`;
  }

  // Convert text to sparse vector for searching
  private tokenizeToVector(text: string): { indices: number[]; values: number[] } {
    const indices = this.tokenizeToIndices(text);
    const values = this.tokenizeToValues(text);
    // Return sparse vector in proper Qdrant format
    return { indices, values };
  }

  // Apply RRF (Reciprocal Rank Fusion) or weighted reranking
  private applyRerankingIfNeeded(
    points: any[],
    rerankStrategy?: { strategy: 'rrf' | 'weighted'; params?: Record<string, any> }
  ): any[] {
    if (!rerankStrategy) {
      return points;
    }

    if (rerankStrategy.strategy === 'rrf') {
      // Simple RRF: normalize scores and combine
      // Since we're working with already-ranked results, just return as-is
      return points.sort((a, b) => b.score - a.score);
    }

    return points;
  }
}
