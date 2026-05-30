const {Dex} = require('../../../engine/pokemon-showdown/.sim-dist/dex');

const moves = Dex.moves.all();
const moveDict = {};

for (const move of moves) {
    moveDict[move.id] = {
        id: move.id,
        name: move.name,
        basePower: move.basePower,
        type: move.type,
        accuracy: move.accuracy === true ? 100 : move.accuracy,
        category: move.category
    };
}

console.log(JSON.stringify(moveDict, null, 2));
