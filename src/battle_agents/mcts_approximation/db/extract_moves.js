const path = require('path');
const fs = require('fs');

const engineDir = path.resolve(__dirname, '../../../../engine');
const { Dex } = require(path.join(engineDir, 'dist/sim/dex'));

const dexGen3 = Dex.mod('gen3');

// Output to the same directory as this script
const outDir = path.resolve(__dirname, '.');

const moves = dexGen3.moves.all();
const moveDict = {};

for (const move of moves) {
    // Only Gen 3 or earlier moves
    if (move.gen > 3) continue;
    
    moveDict[move.id] = {
        id: move.id,
        name: move.name,
        basePower: move.basePower,
        type: move.type,
        accuracy: move.accuracy === true ? 100 : move.accuracy,
        category: move.category
    };
}

fs.writeFileSync(path.join(outDir, 'moves.json'), JSON.stringify(moveDict, null, 2));
console.log('Moves written:', Object.keys(moveDict).length);
