"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var moves_exports = {};
__export(moves_exports, {
  Moves: () => Moves
});
module.exports = __toCommonJS(moves_exports);
const Moves = {
  perishsong: {
    inherit: true,
    onTryMove(attacker, defender, move) {
      let immuneMon = false;
      for (const mon of attacker.side.active) {
        if (this.runEvent("Immunity", mon, attacker, move)) {
          immuneMon = true;
        }
      }
      for (const mon of attacker.side.foe.active) {
        if (this.runEvent("Immunity", mon, attacker, move)) {
          immuneMon = true;
        }
      }
      if (immuneMon) return;
      if (this.format.gameType === "singles") {
        if (attacker.side.pokemonLeft === 1 && attacker.side.foe.pokemonLeft === 1) {
          return false;
        }
      } else if (this.format.gameType === "doubles") {
        if (attacker.side.pokemonLeft === 2 && attacker.side.foe.pokemonLeft === 2) {
          return false;
        }
      }
    }
  }
};
//# sourceMappingURL=moves.js.map
