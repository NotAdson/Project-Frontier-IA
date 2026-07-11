/**
 * Extracts Gen 3 species, items, and abilities from the Showdown Dex and writes
 * them to JSON files used by species_db.py, items_db.py, and abilities_db.py.
 */
const path = require('path');
const fs = require('fs');

const engineDir = path.resolve(__dirname, '../../../../engine');
const { Dex } = require(path.join(engineDir, 'dist/sim/dex'));

const dexGen3 = Dex.mod('gen3');

// Output to the same directory as this script
const outDir = path.resolve(__dirname, '.');

// --- Species ---
const speciesDict = {};
for (const s of dexGen3.species.all()) {
    // Only Gen 3 or earlier species (0-386)
    if (s.num <= 0 || s.gen > 3) continue;
    speciesDict[s.id] = {
        id: s.id,
        name: s.name,
        num: s.num,
        types: s.types,
        baseStats: s.baseStats
    };
}
fs.writeFileSync(path.join(outDir, 'species.json'), JSON.stringify(speciesDict, null, 2));
console.log('Species written:', Object.keys(speciesDict).length);

// --- Items ---
const itemsDict = {};
for (const item of dexGen3.items.all()) {
    // Only Gen 3 or earlier items
    if (item.num < 0 || item.gen > 3) continue;
    itemsDict[item.id] = { id: item.id, name: item.name };
}
fs.writeFileSync(path.join(outDir, 'items.json'), JSON.stringify(itemsDict, null, 2));
console.log('Items written:', Object.keys(itemsDict).length);

// --- Abilities ---
const abilitiesDict = {};
for (const ability of dexGen3.abilities.all()) {
    // Only Gen 3 or earlier abilities
    if (ability.num < 0 || ability.gen > 3) continue;
    abilitiesDict[ability.id] = { id: ability.id, name: ability.name };
}
fs.writeFileSync(path.join(outDir, 'abilities.json'), JSON.stringify(abilitiesDict, null, 2));
console.log('Abilities written:', Object.keys(abilitiesDict).length);
