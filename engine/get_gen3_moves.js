const { Dex } = require('./dist/sim/dex');

const gen3Dex = Dex.mod('gen3');
const moves = gen3Dex.moves.all();
const gen3Moves = moves.filter(m => {
    return m.gen <= 3 && m.exists && !m.isNonstandard;
});

console.log(JSON.stringify(gen3Moves.map(m => ({
    id: m.id,
    name: m.name,
    type: m.type,
    category: m.category, // Physical, Special, Status
    basePower: m.basePower,
    accuracy: m.accuracy === true ? 0 : m.accuracy, // 0 for cannot miss
    priority: m.priority,
    flags: m.flags,
    secondary: m.secondary || (m.secondaries ? m.secondaries[0] : null),
    boosts: m.boosts,
    status: m.status,
    volatileStatus: m.volatileStatus,
    self: m.self,
    heal: m.heal,
    drain: m.drain,
    recoil: m.recoil,
    multihit: m.multihit
})), null, 2));
