/**
 * Script to index all law resources and generate a metadata file
 * Run with: node scripts/indexLawResources.cjs
 */

const fs = require('fs');
const path = require('path');

const LAW_RESOURCES_DIR = path.join(__dirname, '..', 'Law resouces  copy 2');
const OUTPUT_FILE = path.join(__dirname, '..', 'law-resources-index.json');

function walkDir(dir, category = '', subcategory = '') {
  const entries = [];
  
  if (!fs.existsSync(dir)) {
    console.error(`Directory not found: ${dir}`);
    return entries;
  }

  const items = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const item of items) {
    const fullPath = path.join(dir, item.name);
    
    // Skip hidden files and temp files
    if (item.name.startsWith('.') || item.name.startsWith('~$')) {
      continue;
    }
    
    if (item.isDirectory()) {
      const newCategory = category || item.name.replace(' copy', '').trim();
      const newSubcategory = category ? item.name : '';
      entries.push(...walkDir(fullPath, newCategory, newSubcategory));
    } else if (item.name.toLowerCase().endsWith('.pdf')) {
      const stats = fs.statSync(fullPath);
      const relativePath = path.relative(path.join(__dirname, '..'), fullPath);
      
      entries.push({
        id: Buffer.from(relativePath).toString('base64').replace(/[/+=]/g, '').substring(0, 12),
        name: item.name.replace('.pdf', ''),
        path: relativePath,
        category: category || 'General',
        subcategory: subcategory || '',
        mimeType: 'application/pdf',
        size: stats.size
      });
    }
  }
  
  return entries;
}

function main() {
  console.log('🔍 Indexing Law Resources...\n');
  
  const resources = walkDir(LAW_RESOURCES_DIR);
  
  // Extract unique categories
  const categories = [...new Set(resources.map(r => r.category))].sort();
  
  const index = {
    generatedAt: new Date().toISOString(),
    totalFiles: resources.length,
    categories,
    resources
  };
  
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(index, null, 2));
  
  console.log(`✅ Indexed ${resources.length} PDF files`);
  console.log(`📁 Categories: ${categories.join(', ')}`);
  console.log(`💾 Saved to: ${OUTPUT_FILE}`);
  
  // Print category breakdown
  console.log('\n📊 Category breakdown:');
  categories.forEach(cat => {
    const count = resources.filter(r => r.category === cat).length;
    console.log(`   - ${cat}: ${count} files`);
  });
}

main();
