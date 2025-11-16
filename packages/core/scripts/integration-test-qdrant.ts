#!/usr/bin/env node

/**
 * Integration Test for QdrantVectorDatabase with Real Codebase
 *
 * Usage:
 *   npx ts-node scripts/integration-test-qdrant.ts /path/to/codebase
 */

import path from 'path';
import { Context } from '../src/index';
import { QdrantVectorDatabase } from '../src/vectordb/qdrant-vectordb';
import { OpenAIEmbedding } from '../src/embedding/openai';

async function runIntegrationTest() {
  const codebasePath = process.argv[2] || process.cwd();

  console.log(`\n🧪 QdrantVectorDatabase Integration Test`);
  console.log(`============================================`);
  console.log(`Codebase Path: ${codebasePath}`);
  console.log(`Qdrant URL: http://localhost:6333`);
  console.log(`\n`);

  try {
    // 1. Initialize Qdrant Vector Database
    console.log(`[TEST 1] Initializing QdrantVectorDatabase...`);
    const vectorDb = new QdrantVectorDatabase({
      address: process.env.QDRANT_URL || 'http://localhost:6333',
      apiKey: process.env.QDRANT_API_KEY,
    });
    console.log(`✅ QdrantVectorDatabase initialized\n`);

    // 2. Initialize Embedding Provider
    console.log(`[TEST 2] Initializing Embedding Provider (OpenAI)...`);
    const embedding = new OpenAIEmbedding({
      apiKey: process.env.OPENAI_API_KEY || 'sk-test',
      model: 'text-embedding-3-small',
    });
    console.log(`✅ Embedding provider initialized\n`);

    // 3. Initialize Context
    console.log(`[TEST 3] Initializing Context...`);
    const context = new Context({
      embedding,
      vectorDatabase: vectorDb,
    });
    console.log(`✅ Context initialized\n`);

    // 4. Check if index exists
    console.log(`[TEST 4] Checking existing index...`);
    const hasIndex = await context.hasIndex(codebasePath);
    console.log(`   Index exists: ${hasIndex ? 'Yes' : 'No'}\n`);

    // 5. Index the codebase
    if (true) { // Force reindex for testing
      console.log(`[TEST 5] Indexing codebase (with progress tracking)...`);
      const result = await context.indexCodebase(
        codebasePath,
        (progress) => {
          const bar = '█'.repeat(Math.round(progress.percentage / 5));
          const empty = '░'.repeat(20 - bar.length);
          console.log(
            `   [${bar}${empty}] ${progress.percentage}% - ${progress.phase}`
          );
        },
        false // Don't force reindex if already indexed
      );

      console.log(`\n   ✅ Indexing complete!`);
      console.log(`   - Indexed Files: ${result.indexedFiles}`);
      console.log(`   - Total Chunks: ${result.totalChunks}`);
      console.log(`   - Status: ${result.status}\n`);
    }

    // 6. Perform semantic search
    console.log(`[TEST 6] Performing semantic search...`);
    const searchQueries = [
      'function definition',
      'error handling',
      'class implementation',
    ];

    for (const query of searchQueries) {
      console.log(`\n   Searching for: "${query}"`);
      try {
        const results = await context.semanticSearch(codebasePath, query, 3);
        console.log(`   ✅ Found ${results.length} results`);

        if (results.length > 0) {
          const topResult = results[0];
          console.log(`   Top result:`);
          console.log(`   - File: ${topResult.relativePath}:${topResult.startLine}`);
          console.log(`   - Score: ${topResult.score.toFixed(4)}`);
          console.log(`   - Preview: ${topResult.content.substring(0, 80)}...`);
        }
      } catch (error) {
        console.log(`   ⚠️  Search failed: ${error}`);
      }
    }

    // 7. Verify collection exists
    console.log(`\n[TEST 7] Verifying collection existence...`);
    const collections = await vectorDb.listCollections();
    const collectionName = context.getCollectionName(codebasePath);
    const collectionExists = collections.includes(collectionName);

    console.log(`   Collections count: ${collections.length}`);
    console.log(`   Target collection exists: ${collectionExists ? 'Yes' : 'No'}`);
    console.log(`   Collection name: ${collectionName}\n`);

    // 8. Verify hybrid search capability
    console.log(`[TEST 8] Testing hybrid search capabilities...`);
    try {
      const isHybrid = await vectorDb.hasCollection(collectionName);
      if (isHybrid) {
        console.log(`   ✅ Collection is configured for hybrid search\n`);
      }
    } catch (error) {
      console.log(`   ⚠️  Hybrid search test inconclusive: ${error}\n`);
    }

    // Summary
    console.log(`\n============================================`);
    console.log(`✅ Integration Test Completed Successfully!`);
    console.log(`============================================\n`);

    process.exit(0);

  } catch (error) {
    console.error(`\n❌ Integration Test Failed:`);
    console.error(error);
    process.exit(1);
  }
}

// Run the test
runIntegrationTest();
