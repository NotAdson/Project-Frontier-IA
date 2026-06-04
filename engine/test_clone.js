const { Battle } = require('./dist/sim/battle');
const { Teams } = require('./dist/sim/teams');

function benchmark() {
    const format = 'gen3randombattle';
    const p1_team = Teams.pack(Teams.generate(format));
    const p2_team = Teams.pack(Teams.generate(format));
    
    const battle = new Battle({
        formatid: format,
        send: () => {},
    });
    battle.setPlayer('p1', { name: 'Player 1', team: p1_team });
    battle.setPlayer('p2', { name: 'Player 2', team: p2_team });
    
    // Warm up
    let state = battle.toJSON();
    let b2 = Battle.fromJSON(state);
    
    console.log("Measuring 500 clone iterations in Node.js...");
    const start = Date.now();
    for (let i = 0; i < 500; i++) {
        const s = b2.toJSON();
        b2 = Battle.fromJSON(s);
        b2.choose('p1', 'default');
        b2.choose('p2', 'default');
    }
    const end = Date.now();
    const duration = end - start;
    console.log(`Total time: ${duration} ms`);
    console.log(`Average clone + step time: ${duration / 500} ms`);
}

benchmark();
