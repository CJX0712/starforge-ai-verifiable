/**
 * @file tools/gen-data.mjs — 由 data/ 原始数据生成浏览器友好的 ES 数据模块
 * 作者：晨星
 *
 * 数据来源（均为公有领域 / 公开数据集）：
 * - data/iris.data：UCI Machine Learning Repository，R.A. Fisher (1936)，CC BY 4.0
 * - data/wine.data：UCI Machine Learning Repository，Forina et al.（葡萄酒化学分析）
 * - data/corpus.txt：Project Gutenberg 公有领域文本
 *   （Alice's Adventures in Wonderland, Lewis Carroll + The Complete Works of William Shakespeare）
 *
 * 运行：node tools/gen-data.mjs
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

// ---- Iris（鸢尾花，Fisher 1936）----
const irisLines = readFileSync(join(root, 'data/iris.data'), 'utf8').split('\n').filter((l) => l.trim());
const irisRows = irisLines.map((l) => {
  const p = l.split(',');
  return [+p[0], +p[1], +p[2], +p[3], p[4]];
});
writeFileSync(
  join(root, 'data/iris.mjs'),
  '// 由 UCI iris.data 生成（R.A. Fisher, 1936；CC BY 4.0）\n' +
    'export const IRIS_FEATURES = ' +
    JSON.stringify(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']) +
    ';\nexport const IRIS_ROWS = ' +
    JSON.stringify(irisRows) +
    ';\n'
);

// ---- Wine（葡萄酒，UCI）----
const wineLines = readFileSync(join(root, 'data/wine.data'), 'utf8').split('\n').filter((l) => l.trim());
const wineRows = wineLines.map((l) => l.split(',').map(Number));
const wineFeatureNames = [
  'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
  'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
  'color_intensity', 'hue', 'od280_od315_of_diluted_wines', 'proline',
];
writeFileSync(
  join(root, 'data/wine.mjs'),
  '// 由 UCI wine.data 生成（Forina et al.；CC BY 4.0）\n' +
    'export const WINE_FEATURES = ' +
    JSON.stringify(wineFeatureNames) +
    ';\nexport const WINE_ROWS = ' +
    JSON.stringify(wineRows) +
    ';\n'
);

// ---- 字符级语料 ----
const text = readFileSync(join(root, 'data/corpus.txt'), 'utf8');
writeFileSync(
  join(root, 'data/corpus.mjs'),
  "// 由 Project Gutenberg 公有领域文本生成：\n" +
    "// Alice's Adventures in Wonderland (Lewis Carroll) + The Complete Works of William Shakespeare\n" +
    'export const CORPUS_TEXT = ' +
    JSON.stringify(text) +
    ';\n'
);

console.log(`数据模块生成完成：iris ${irisRows.length} 行，wine ${wineRows.length} 行，corpus ${text.length} 字符`);
