// run-battle.js  — minimal Gen 3 singles runner (async-iterator style)
const {BattleStream} = require('./dist/sim');
const readline = require('readline');

// create the battle stream
const stream = new BattleStream();

// ---------- task 1: pipe every message the simulator produces to stdout
(async () => {
  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
})();

// ---------- task 2: feed commands from our own stdin into the simulator
(async () => {
  const format = process.argv[2] || 'gen3ou';
  stream.write(`>start {"formatid":"${format}"}`);

  const rl = readline.createInterface({input: process.stdin});
  for await (const line of rl) {
    stream.write(line);
  }
})();
