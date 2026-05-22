const { Battle, extractChannelMessages } = require('./dist/sim/battle');
const { Teams } = require('./dist/sim/teams');
const readline = require('readline');

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

console.log(JSON.stringify({ type: "ready" }));

rl.on('line', (line) => {
    if (!line.trim()) return;
    try {
        const request = JSON.parse(line);
        if (request.type === 'init') {
            const format = request.formatid || 'gen3randombattle';
            // Use random teams if none provided
            const p1_team = request.p1_team || Teams.pack(Teams.generate(format));
            const p2_team = request.p2_team || Teams.pack(Teams.generate(format));
            
            const battle = new Battle({
                formatid: format,
                send: () => {}, // Disable default logging to avoid polluting stdout
            });
            
            battle.setPlayer('p1', { name: 'Player 1', team: p1_team });
            battle.setPlayer('p2', { name: 'Player 2', team: p2_team });
            
            const response = {
                type: "success", 
                state: battle.toJSON(),
                request: battle.p1.activeRequest,
                p2_request: battle.p2.activeRequest,
                winner: battle.winner,
                log: extractChannelMessages(battle.log.join('\n'), [0])[0]
            };
            console.log(JSON.stringify(response));
            
        } else if (request.type === 'result') {
            const battle = Battle.fromJSON(request.state);
            battle.send = () => {};
            
            // Choose actions
            if (request.p1_action) battle.choose('p1', request.p1_action);
            if (request.p2_action) {
                battle.choose('p2', request.p2_action);
            } else {
                battle.choose('p2', 'default'); // Default random action for opponent
            }
            
            const response = {
                type: "success", 
                state: battle.toJSON(),
                request: battle.p1.activeRequest,
                p2_request: battle.p2.activeRequest,
                winner: battle.winner,
                log: extractChannelMessages(battle.log.join('\n'), [0])[0]
            };
            console.log(JSON.stringify(response));
        }
    } catch (e) {
        console.log(JSON.stringify({ type: "error", error: e.message, stack: e.stack }));
    }
});
