const { Battle } = require('./dist/sim/battle');
const { Teams } = require('./dist/sim/teams');

const format = 'gen3randombattle';
const p1_team = Teams.pack(Teams.generate(format));
const p2_team = Teams.pack(Teams.generate(format));

const battle = new Battle({
    formatid: format,
    send: () => {}, 
});

battle.setPlayer('p1', { name: 'Player 1', team: p1_team });
battle.setPlayer('p2', { name: 'Player 2', team: p2_team });
battle.makeChoices('move 1', 'move 1');

// See if we can extract spectator log
let spectatorLog = battle.log;
try {
    const { extractChannelMessages } = require('./dist/sim/battle');
    if (extractChannelMessages) {
        spectatorLog = extractChannelMessages(battle.log.join('\n'), [0])[0];
        console.log("Extracted successfully! Length: " + spectatorLog.length);
        console.log("Has split? " + spectatorLog.join('\n').includes('|split|'));
    } else {
        console.log("extractChannelMessages not found");
    }
} catch(e) {
    console.log("Error: " + e.message);
}

