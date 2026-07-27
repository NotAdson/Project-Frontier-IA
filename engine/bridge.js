const { Battle, extractChannelMessages } = require('./dist/sim/battle');
const { Teams } = require('./dist/sim/teams');
const readline = require('readline');

function getValidActions(battle, player) {
    const request = player === 'p1' ? battle.p1.activeRequest : battle.p2.activeRequest;
    if (!request) return ["pass"];
    
    const actions = [];
    if (request.forceSwitch) {
        const side = request.side || {};
        const pokemon = side.pokemon || [];
        for (let i = 0; i < pokemon.length; i++) {
            const p = pokemon[i];
            if (!p.active && p.condition !== '0 fnt') {
                actions.push(`switch ${i + 1}`);
            }
        }
    } else if (request.active) {
        const moves = (request.active[0] && request.active[0].moves) || [];
        for (let i = 0; i < moves.length; i++) {
            if (!moves[i].disabled) {
                actions.push(`move ${i + 1}`);
            }
        }
        
        const side = request.side || {};
        const trapped = (request.active[0] && request.active[0].trapped) || false;
        if (!trapped) {
            const pokemon = side.pokemon || [];
            for (let i = 0; i < pokemon.length; i++) {
                const p = pokemon[i];
                if (!p.active && p.condition !== '0 fnt') {
                    actions.push(`switch ${i + 1}`);
                }
            }
        }
    }
    
    if (actions.length === 0) {
        actions.push("pass");
    }
    return actions;
}

// Cache settings to avoid memory leaks
const MAX_CACHE_SIZE = 500;
const stateCache = new Map();
const stateIdQueue = [];
let nextStateId = 1;

function cacheState(battle, serializedState) {
    const id = nextStateId++;
    // Store as string to prevent shared reference mutations across branching search paths
    const serializedStateStr = JSON.stringify(serializedState);
    stateCache.set(id, { battle, serializedStateStr });
    stateIdQueue.push(id);
    
    if (stateCache.size > MAX_CACHE_SIZE) {
        const oldestId = stateIdQueue.shift();
        stateCache.delete(oldestId);
    }
    return id;
}

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
            const format = request.formatid || 'gen3ou';

            // Parse team strings into team objects so setPlayer/getTeam skips
            // Teams.unpack (which rejects the multi-line format used in
            // data/teams/gen3ou.txt). Teams.import accepts both the multi-line
            // Showdown text format and the packed '|' format.
            let p1_team, p2_team;
            if (request.p1_team) {
                p1_team = Teams.import(request.p1_team);
            }
            if (!p1_team || p1_team.length === 0) p1_team = Teams.generate(format);

            if (request.p2_team) {
                p2_team = Teams.import(request.p2_team);
            }
            if (!p2_team || p2_team.length === 0) p2_team = Teams.generate(format);

            const battle = new Battle({
                formatid: format,
                send: () => {}, // Disable default logging to avoid polluting stdout
            });
            
            battle.setPlayer('p1', { name: 'Player 1', team: p1_team });
            battle.setPlayer('p2', { name: 'Player 2', team: p2_team });
            
            const serializedState = battle.toJSON();
            const stateId = cacheState(battle, serializedState);
            
            const response = {
                type: "success", 
                state_id: stateId,
                state: serializedState,
                request: battle.p1.activeRequest,
                p2_request: battle.p2.activeRequest,
                winner: battle.winner,
                log: extractChannelMessages(battle.log.join('\n'), [0])[0]
            };
            console.log(JSON.stringify(response));
            
        } else if (request.type === 'result') {
            let battle;
            if (request.state_id !== undefined && stateCache.has(request.state_id)) {
                const cached = stateCache.get(request.state_id);
                // Parse the stringified cached state to ensure a clean copy
                battle = Battle.fromJSON(JSON.parse(cached.serializedStateStr));
            } else {
                // Fallback to slower pipe deserialization
                battle = Battle.fromJSON(request.state);
            }
            battle.send = () => {};
            
            // Choose actions
            if (request.p1_action) battle.choose('p1', request.p1_action);
            if (request.p2_action) {
                battle.choose('p2', request.p2_action);
            } else {
                battle.choose('p2', 'default'); // Default random action for opponent
            }
            
            const serializedState = battle.toJSON();
            const stateId = cacheState(battle, serializedState);
            
            const response = {
                type: "success", 
                state_id: stateId,
                state: serializedState,
                request: battle.p1.activeRequest,
                p2_request: battle.p2.activeRequest,
                winner: battle.winner,
                log: extractChannelMessages(battle.log.join('\n'), [0])[0]
            };
            console.log(JSON.stringify(response));
        } else if (request.type === 'rollout') {
            let battle;
            if (request.state_id !== undefined && stateCache.has(request.state_id)) {
                battle = Battle.fromJSON(JSON.parse(stateCache.get(request.state_id).serializedStateStr));
            } else {
                battle = Battle.fromJSON(request.state);
            }
            battle.send = () => {};
            const player = request.player;
            const maxDepth = request.max_depth || 150;
            
            let depth = 0;
            while (!battle.winner && depth < maxDepth) {
                const actions = getValidActions(battle, player);
                const action = actions[Math.floor(Math.random() * actions.length)];
                
                if (player === 'p1') {
                    battle.choose('p1', action);
                    battle.choose('p2', 'default');
                } else {
                    battle.choose('p1', 'default');
                    battle.choose('p2', action);
                }
                depth++;
            }
            
            let reward = 0.5;
            if (battle.winner) {
                if (battle.winner === 'Player 1' && player === 'p1') reward = 1.0;
                else if (battle.winner === 'Player 2' && player === 'p2') reward = 1.0;
                else reward = 0.0;
            }
            
            const response = {
                type: "success",
                reward: reward
            };
            console.log(JSON.stringify(response));
        } else if (request.type === 'clear_cache') {
            stateCache.clear();
            stateIdQueue.length = 0;
            console.log(JSON.stringify({ type: "success" }));
        }
    } catch (e) {
        console.log(JSON.stringify({ type: "error", error: e.message, stack: e.stack }));
    }
});
