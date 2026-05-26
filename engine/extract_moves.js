const { Dex } = require('./dist/sim/dex');
const fs = require('fs');

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

fs.writeFileSync('../src/agents/mcts_approximation/moves.json', JSON.stringify(moveDict, null, 2));
console.log("Done");
