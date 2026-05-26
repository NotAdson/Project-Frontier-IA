/**
 * Extracts species, items, and abilities from the Showdown Dex and writes
 * them to JSON files used by species_db.py, items_db.py, and abilities_db.py.
 *
 * Must be run from the engine/ directory:
 *   cd engine && node ../src/agents/mcts_approximation/extract_dex.js
 */
const { Dex } = require('./dist/sim/dex');
const fs = require('fs');
const path = require('path');

// Output to the same directory as this script
const outDir = path.resolve(__dirname, '.');

// --- Species ---
const speciesDict = {};
for (const s of Dex.species.all()) {
    if (s.num <= 0) continue;
    speciesDict[s.id] = { id: s.id, name: s.name, num: s.num };
}
fs.writeFileSync(path.join(outDir, 'species.json'), JSON.stringify(speciesDict));
console.log('Species written:', Object.keys(speciesDict).length);

// --- Items ---
const itemsDict = {};
for (const item of Dex.items.all()) {
    if (item.num < 0) continue;
    itemsDict[item.id] = { id: item.id, name: item.name };
}
fs.writeFileSync(path.join(outDir, 'items.json'), JSON.stringify(itemsDict));
console.log('Items written:', Object.keys(itemsDict).length);

// --- Abilities ---
const abilitiesDict = {};
for (const ability of Dex.abilities.all()) {
    if (ability.num < 0) continue;
    abilitiesDict[ability.id] = { id: ability.id, name: ability.name };
}
fs.writeFileSync(path.join(outDir, 'abilities.json'), JSON.stringify(abilitiesDict));
console.log('Abilities written:', Object.keys(abilitiesDict).length);
