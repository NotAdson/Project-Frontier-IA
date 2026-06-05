const { Dex } = require('./dist/sim/dex');
const { Teams } = require('./dist/sim/teams');

function generateTeamWithStats(format) {
    const team = Teams.generate(format);
    const result = [];
    for (const p of team) {
        const species = Dex.species.get(p.species);
        const types = species.types;
        const baseStats = species.baseStats;
        
        // Calculate stats
        const stats = {};
        const ivs = p.ivs || { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
        const evs = p.evs || { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 };
        
        // HP
        const hpIv = ivs.hp !== undefined ? ivs.hp : 31;
        const hpEv = evs.hp !== undefined ? evs.hp : 85;
        stats.hp = Math.floor((2 * baseStats.hp + hpIv + Math.floor(hpEv / 4)) * p.level / 100) + p.level + 10;
        
        // Others
        for (const stat of ['atk', 'def', 'spa', 'spd', 'spe']) {
            const iv = ivs[stat] !== undefined ? ivs[stat] : 31;
            const ev = evs[stat] !== undefined ? evs[stat] : 85;
            stats[stat] = Math.floor((2 * baseStats[stat] + iv + Math.floor(ev / 4)) * p.level / 100) + 5;
        }
        
        result.push({
            species_id: p.species.toLowerCase().replace(/[^a-z0-9]/g, ''),
            name: p.name,
            level: p.level,
            hp: stats.hp,
            max_hp: stats.hp,
            atk: stats.atk,
            def: stats.def,
            spa: stats.spa,
            spd: stats.spd,
            spe: stats.spe,
            type1: types[0] || 'None',
            type2: types[1] || 'None',
            ability: p.ability,
            item: p.item,
            moves: p.moves
        });
    }
    return result;
}

if (require.main === module) {
    const format = process.argv[2] || 'gen3randombattle';
    const p1 = generateTeamWithStats(format);
    const p2 = generateTeamWithStats(format);
    console.log(JSON.stringify({ p1, p2 }));
}
