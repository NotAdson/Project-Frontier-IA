const { Dex } = require('./dist/sim/dex');

// Set mod to gen3
const gen3Dex = Dex.mod('gen3');

const abilities = gen3Dex.abilities.all();
const gen3Abilities = abilities.filter(a => {
    return a.gen <= 3 && a.exists && !a.isNonstandard;
});

console.log(JSON.stringify(gen3Abilities.map(a => ({ id: a.id, name: a.name, desc: a.desc || a.shortDesc, gen: a.gen })), null, 2));
