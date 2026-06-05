const { Dex } = require('./dist/sim/dex');
const fs = require('fs');

const speciesList = Dex.species.all();
const learnsetDict = {};

for (const species of speciesList) {
    if (!species.id) continue;
    try {
        const fullLearnsetList = Dex.species.getFullLearnset(species.id);
        if (fullLearnsetList && fullLearnsetList.length > 0) {
            const learnsetObj = fullLearnsetList[0].learnset;
            if (learnsetObj) {
                // Keep only the move keys
                learnsetDict[species.id] = Object.keys(learnsetObj);
            }
        }
    } catch (e) {
        console.log("Error extracting for " + species.id + ": " + e.message);
    }
}

fs.writeFileSync('/home/adson/Codes/University/IA/Pokemon/src/battle_agents/mcts_approximation/db/learnsets.json', JSON.stringify(learnsetDict, null, 2));
console.log("Extracted learnsets for " + Object.keys(learnsetDict).length + " species.");
