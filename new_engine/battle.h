#ifndef BATTLE_H
#define BATTLE_H

#include "types.h"
#include "pokemon.h"
#include "move.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>

struct Side {
    std::string name;
    std::vector<Pokemon> team;
    int active_idx = 0;
    int spikes_layers = 0;

    Pokemon& get_active() {
        return team[active_idx];
    }

    const Pokemon& get_active() const {
        return team[active_idx];
    }

    bool has_usable_pokemon() const {
        for (const auto& p : team) {
            if (!p.is_fainted()) return true;
        }
        return false;
    }
};

class Battle {
public:
    Side p1;
    Side p2;
    static constexpr int MAX_TURNS = 1000;
    int turn_count = 1;
    std::string winner = "";
    Weather weather = Weather::NONE;
    int weather_turns = 0;
    std::mt19937 rng;

    Battle(Side player1, Side player2, unsigned int seed = 1337) 
        : p1(player1), p2(player2), rng(seed) {
        // Apply start-of-battle Intimidate & weather summoning
        trigger_switch_in_effects(p1, p2);
        trigger_switch_in_effects(p2, p1);
    }

    Weather get_active_weather() const {
        // Cloud Nine and Air Lock negate all weather effects
        if (p1.get_active().ability == "Cloud Nine" || p1.get_active().ability == "Air Lock" ||
            p2.get_active().ability == "Cloud Nine" || p2.get_active().ability == "Air Lock") {
            return Weather::NONE;
        }
        return weather;
    }

    void trigger_switch_in_effects(Side& switching_side, Side& opponent_side) {
        Pokemon& active = switching_side.get_active();
        Pokemon& opponent = opponent_side.get_active();
        if (active.is_fainted()) return;

        // Trace copies opponent's ability
        if (active.ability == "Trace" && !opponent.is_fainted() && !opponent.ability.empty() && opponent.ability != "Trace") {
            active.ability = opponent.ability;
        }

        // Intimidate (blocked by substitute)
        if (active.ability == "Intimidate" && !opponent.is_fainted() && opponent.substitute_hp == 0) {
            opponent.apply_boost(opponent.boost_atk, -1);
        }

        // Weather summoning abilities (permanent weather in Gen 3)
        if (active.ability == "Drizzle") {
            weather = Weather::RAIN;
            weather_turns = -1;
        } else if (active.ability == "Drought") {
            weather = Weather::SUN;
            weather_turns = -1;
        } else if (active.ability == "Sand Stream") {
            weather = Weather::SANDSTORM;
            weather_turns = -1;
        }

        // Forecast Castform type updates
        update_forecast(switching_side, get_active_weather());
    }

    void update_forecast(Side& side, Weather current_weather) {
        Pokemon& p = side.get_active();
        if (p.ability == "Forecast" && p.id == "castform") {
            if (current_weather == Weather::SUN) p.type1 = Type::FIRE;
            else if (current_weather == Weather::RAIN) p.type1 = Type::WATER;
            else if (current_weather == Weather::HAIL) p.type1 = Type::ICE;
            else p.type1 = Type::NORMAL;
        }
    }

    bool is_trapped(const Pokemon& p, const Pokemon& opp) const {
        if (opp.is_fainted()) return false;
        
        // Shadow Tag prevents switching
        if (opp.ability == "Shadow Tag") return true;
        
        // Arena Trap prevents non-airborne switches
        if (opp.ability == "Arena Trap" && p.type1 != Type::FLYING && p.type2 != Type::FLYING && p.ability != "Levitate") {
            return true;
        }

        // Magnet Pull traps Steel types
        if (opp.ability == "Magnet Pull" && (p.type1 == Type::STEEL || p.type2 == Type::STEEL)) {
            return true;
        }

        return false;
    }

    // Gen 3 Type Effectiveness Chart Lookup
    static float get_type_effectiveness(Type atk, Type def) {
        if (atk == Type::NONE || def == Type::NONE) return 1.0f;
        
        switch(atk) {
            case Type::NORMAL:
                if (def == Type::ROCK || def == Type::STEEL) return 0.5f;
                if (def == Type::GHOST) return 0.0f;
                break;
            case Type::FIRE:
                if (def == Type::GRASS || def == Type::ICE || def == Type::BUG || def == Type::STEEL) return 2.0f;
                if (def == Type::FIRE || def == Type::WATER || def == Type::ROCK || def == Type::DRAGON) return 0.5f;
                break;
            case Type::WATER:
                if (def == Type::FIRE || def == Type::GROUND || def == Type::ROCK) return 2.0f;
                if (def == Type::WATER || def == Type::GRASS || def == Type::DRAGON) return 0.5f;
                break;
            case Type::GRASS:
                if (def == Type::WATER || def == Type::GROUND || def == Type::ROCK) return 2.0f;
                if (def == Type::FIRE || def == Type::GRASS || def == Type::POISON || def == Type::FLYING || def == Type::BUG || def == Type::DRAGON || def == Type::STEEL) return 0.5f;
                break;
            case Type::ELECTRIC:
                if (def == Type::WATER || def == Type::FLYING) return 2.0f;
                if (def == Type::ELECTRIC || def == Type::GRASS || def == Type::DRAGON) return 0.5f;
                if (def == Type::GROUND) return 0.0f;
                break;
            case Type::ICE:
                if (def == Type::GRASS || def == Type::GROUND || def == Type::FLYING || def == Type::DRAGON) return 2.0f;
                if (def == Type::FIRE || def == Type::WATER || def == Type::ICE || def == Type::STEEL) return 0.5f;
                break;
            case Type::FIGHTING:
                if (def == Type::NORMAL || def == Type::ICE || def == Type::ROCK || def == Type::STEEL || def == Type::DARK) return 2.0f;
                if (def == Type::POISON || def == Type::FLYING || def == Type::PSYCHIC || def == Type::BUG) return 0.5f;
                if (def == Type::GHOST) return 0.0f;
                break;
            case Type::POISON:
                if (def == Type::GRASS) return 2.0f;
                if (def == Type::POISON || def == Type::GROUND || def == Type::ROCK || def == Type::GHOST) return 0.5f;
                if (def == Type::STEEL) return 0.0f;
                break;
            case Type::GROUND:
                if (def == Type::FIRE || def == Type::ELECTRIC || def == Type::POISON || def == Type::ROCK || def == Type::STEEL) return 2.0f;
                if (def == Type::GRASS || def == Type::BUG) return 0.5f;
                if (def == Type::FLYING) return 0.0f;
                break;
            case Type::FLYING:
                if (def == Type::GRASS || def == Type::FIGHTING || def == Type::BUG) return 2.0f;
                if (def == Type::ELECTRIC || def == Type::ROCK || def == Type::STEEL) return 0.5f;
                break;
            case Type::PSYCHIC:
                if (def == Type::FIGHTING || def == Type::POISON) return 2.0f;
                if (def == Type::PSYCHIC || def == Type::STEEL) return 0.5f;
                if (def == Type::DARK) return 0.0f;
                break;
            case Type::BUG:
                if (def == Type::GRASS || def == Type::PSYCHIC || def == Type::DARK) return 2.0f;
                if (def == Type::FIRE || def == Type::FIGHTING || def == Type::POISON || def == Type::FLYING || def == Type::GHOST || def == Type::STEEL) return 0.5f;
                break;
            case Type::ROCK:
                if (def == Type::FIRE || def == Type::ICE || def == Type::FLYING || def == Type::BUG) return 2.0f;
                if (def == Type::FIGHTING || def == Type::GROUND || def == Type::STEEL) return 0.5f;
                break;
            case Type::GHOST:
                if (def == Type::PSYCHIC || def == Type::GHOST) return 2.0f;
                if (def == Type::DARK || def == Type::STEEL) return 0.5f;
                if (def == Type::NORMAL) return 0.0f;
                break;
            case Type::DRAGON:
                if (def == Type::DRAGON) return 2.0f;
                if (def == Type::STEEL) return 0.5f;
                break;
            case Type::STEEL:
                if (def == Type::ICE || def == Type::ROCK) return 2.0f;
                if (def == Type::FIRE || def == Type::WATER || def == Type::ELECTRIC || def == Type::STEEL) return 0.5f;
                break;
            case Type::DARK:
                if (def == Type::PSYCHIC || def == Type::GHOST) return 2.0f;
                if (def == Type::FIGHTING || def == Type::DARK || def == Type::STEEL) return 0.5f;
                break;
            default:
                break;
        }
        return 1.0f;
    }

    static float get_total_effectiveness(Type move_type, const Pokemon& target) {
        float eff = get_type_effectiveness(move_type, target.type1);
        if (target.type2 != Type::NONE) {
            eff *= get_type_effectiveness(move_type, target.type2);
        }
        return eff;
    }

    // Standard Gen 3 Damage Formula
    static int calculate_damage(const Pokemon& attacker, const Pokemon& defender, const Move& move, float random_factor = 1.0f, Weather weather = Weather::NONE, bool is_crit = false) {
        if (move.category == Category::STATUS || move.base_power == 0) return 0;

        // Battle Armor & Shell Armor block critical hits
        if (defender.ability == "Battle Armor" || defender.ability == "Shell Armor") {
            is_crit = false;
        }

        // Wonder Guard immune to non-supereffective moves
        if (defender.ability == "Wonder Guard") {
            if (get_total_effectiveness(move.type, defender) <= 1.0f) {
                return 0; // immune
            }
        }

        // 1. Level factor
        double lvl_factor = (2.0 * attacker.level / 5.0) + 2.0;

        // 2. Attack and Defense values (applying boosts, items, abilities, and ignoring stages on crit)
        int atk_val = 0;
        int def_val = 0;

        if (move.category == Category::PHYSICAL) {
            int atk_stage = attacker.boost_atk;
            if (is_crit && atk_stage < 0) atk_stage = 0; // crit ignores negative atk stages
            
            // defender's Marvel Scale boosts Defense by 1.5x if status'd
            int def_stage = defender.boost_def;
            if (is_crit && def_stage > 0) def_stage = 0; // crit ignores positive def stages
            
            atk_val = attacker.get_modified_atk(); // applies Huge Power, Hustle, Choice Band, Guts internally
            def_val = defender.get_modified_def(); // applies Marvel Scale internally
            
            // Reapply crit stage ignorings since get_modified uses active boosts
            if (is_crit) {
                if (attacker.boost_atk < 0) {
                    double standard_atk_mul = Pokemon::get_boost_multiplier(0);
                    if (attacker.status != Status::NONE && attacker.ability == "Guts") standard_atk_mul *= 1.5;
                    if (attacker.ability == "Huge Power" || attacker.ability == "Pure Power") standard_atk_mul *= 2.0;
                    if (attacker.ability == "Hustle") standard_atk_mul *= 1.5;
                    if (attacker.item == "Choice Band") standard_atk_mul *= 1.5;
                    atk_val = std::floor(attacker.atk * standard_atk_mul);
                }
                if (defender.boost_def > 0) {
                    double standard_def_mul = Pokemon::get_boost_multiplier(0);
                    if (defender.status != Status::NONE && defender.ability == "Marvel Scale") standard_def_mul *= 1.5;
                    def_val = std::floor(defender.def * standard_def_mul);
                }
            }

            // Burn condition reduces physical attack by 50% (ignored by Guts)
            if (attacker.status == Status::BRN && attacker.ability != "Guts") {
                atk_val = std::floor(atk_val * 0.5);
            }
        } else {
            int spa_stage = attacker.boost_spa;
            if (is_crit && spa_stage < 0) spa_stage = 0;
            atk_val = std::floor(attacker.spa * Pokemon::get_boost_multiplier(spa_stage));

            int spd_stage = defender.boost_spd;
            if (is_crit && spd_stage > 0) spd_stage = 0;
            def_val = defender.get_modified_spd(weather); // applies boosts internally
            def_val = std::floor(def_val * Pokemon::get_boost_multiplier(spd_stage));
        }

        // Avoid division by zero
        if (def_val <= 0) def_val = 1;

        // 3. Base damage
        double base_damage = std::floor(std::floor(lvl_factor * move.base_power * atk_val / def_val) / 50.0) + 2.0;

        // 4. STAB (Same Type Attack Bonus)
        double stab = 1.0;
        if (move.type == attacker.type1 || move.type == attacker.type2) {
            stab = 1.5;
        }

        // 5. Type effectiveness
        double effectiveness = get_total_effectiveness(move.type, defender);
        
        // Levitate immunity to Ground
        if (move.type == Type::GROUND && defender.ability == "Levitate") {
            effectiveness = 0.0f;
        }

        // 6. Weather damage multiplier
        double weather_mul = 1.0;
        if (weather == Weather::SUN) {
            if (move.type == Type::FIRE) weather_mul = 1.5;
            else if (move.type == Type::WATER) weather_mul = 0.5;
        } else if (weather == Weather::RAIN) {
            if (move.type == Type::WATER) weather_mul = 1.5;
            else if (move.type == Type::FIRE) weather_mul = 0.5;
        }

        // 7. Flash Fire damage boost
        if (attacker.ability == "Flash Fire" && attacker.flash_fire_active && move.type == Type::FIRE) {
            base_damage = std::floor(base_damage * 1.5);
        }

        // 8. Thick Fat damage halving
        if (defender.ability == "Thick Fat" && (move.type == Type::FIRE || move.type == Type::ICE)) {
            base_damage = std::floor(base_damage * 0.5);
        }

        // 9. Pinch boosts (Overgrow, Blaze, Torrent, Swarm)
        if (attacker.hp <= attacker.max_hp / 3) {
            if ((attacker.ability == "Overgrow" && move.type == Type::GRASS) ||
                (attacker.ability == "Blaze" && move.type == Type::FIRE) ||
                (attacker.ability == "Torrent" && move.type == Type::WATER) ||
                (attacker.ability == "Swarm" && move.type == Type::BUG)) {
                base_damage = std::floor(base_damage * 1.5);
            }
        }

        // Calculate final damage
        if (effectiveness == 0.0f) return 0;
        double final_damage = base_damage * stab * effectiveness * weather_mul * (is_crit ? 2.0 : 1.0) * random_factor;
        return std::max(1, static_cast<int>(std::floor(final_damage)));
    }

    void apply_spikes_damage(Side& side) {
        Pokemon& p = side.get_active();
        if (p.is_fainted() || side.spikes_layers == 0) return;
        
        // Flying type & Levitate ability are immune to Spikes
        if (p.type1 == Type::FLYING || p.type2 == Type::FLYING || p.ability == "Levitate") return;
        
        double dmg_factor = 0.0;
        if (side.spikes_layers == 1) dmg_factor = 1.0 / 8.0;
        else if (side.spikes_layers == 2) dmg_factor = 1.0 / 6.0;
        else if (side.spikes_layers == 3) dmg_factor = 1.0 / 4.0;
        
        int damage = std::floor(p.max_hp * dmg_factor);
        p.hp = std::max(0, p.hp - damage);
        if (p.hp == 0) {
            p.status = Status::FNT;
        }
    }

    void execute_switch(Side& side, int target_idx) {
        if (target_idx < 0 || target_idx >= static_cast<int>(side.team.size())) return;
        if (side.team[target_idx].is_fainted()) return;

        // Switch Out trigger (Natural Cure)
        Pokemon& old_active = side.get_active();
        if (old_active.ability == "Natural Cure") {
            old_active.status = Status::NONE;
        }
        old_active.reset_volatiles();

        // Switch active index
        side.active_idx = target_idx;

        // Switch In trigger (Trace, Intimidate, Weather, Forecast)
        Side& opponent_side = ( &side == &p1 ) ? p2 : p1;
        trigger_switch_in_effects(side, opponent_side);

        // Apply Spikes damage
        apply_spikes_damage(side);
    }

    // Performs a forced switch-in when a Pokemon has fainted
    void perform_forced_switch(Side& side, int target_idx) {
        execute_switch(side, target_idx);
    }

    // Helper to verify status application immunities
    static bool can_be_statused(Status st, const Pokemon& target) {
        if (st == Status::PAR && target.ability == "Limber") return false;
        if (st == Status::SLP && (target.ability == "Insomnia" || target.ability == "Vital Spirit")) return false;
        if (st == Status::TOX && target.ability == "Immunity") return false;
        if (st == Status::FRZ && target.ability == "Magma Armor") return false;
        if (st == Status::BRN && target.ability == "Water Veil") return false;
        return true;
    }

    // Run one complete turn of the battle
    void run_turn(Action a1, Action a2) {
        if (!winner.empty()) return;

        // --- TURN LIMIT: declare a draw after 1000 turns ---
        if (turn_count >= MAX_TURNS) {
            winner = "Tie";
            return;
        }

        // --- STEP 1: Handle Trapping switch verification ---
        bool p1_switch_trapped = (a1.type == ActionType::SWITCH && is_trapped(p1.get_active(), p2.get_active()));
        bool p2_switch_trapped = (a2.type == ActionType::SWITCH && is_trapped(p2.get_active(), p1.get_active()));

        if (p1_switch_trapped) {
            a1.type = ActionType::PASS; // switch fails
        }
        if (p2_switch_trapped) {
            a2.type = ActionType::PASS;
        }

        // --- STEP 2: Handle Switches First (sorted by speed if both switch) ---
        bool p1_switch = (a1.type == ActionType::SWITCH);
        bool p2_switch = (a2.type == ActionType::SWITCH);

        if (p1_switch && p2_switch) {
            int p1_speed = p1.get_active().get_modified_spe(get_active_weather());
            int p2_speed = p2.get_active().get_modified_spe(get_active_weather());
            if (p1_speed >= p2_speed) {
                execute_switch(p1, a1.index);
                execute_switch(p2, a2.index);
            } else {
                execute_switch(p2, a2.index);
                execute_switch(p1, a1.index);
            }
        } else {
            if (p1_switch) execute_switch(p1, a1.index);
            if (p2_switch) execute_switch(p2, a2.index);
        }

        // Reset Protect flags at the start of move phase
        p1.get_active().is_protected = false;
        p2.get_active().is_protected = false;

        // --- STEP 3: Execute Moves (if applicable) ---
        bool p1_moves = (a1.type == ActionType::MOVE);
        bool p2_moves = (a2.type == ActionType::MOVE);

        // Reset consecutive protect counters if they didn't attempt protect
        if (p1_moves) {
            const Move& m = p1.get_active().moves[a1.index];
            if (m.effect != MoveEffect::PROTECT) p1.get_active().protect_consecutive = 0;
        } else {
            p1.get_active().protect_consecutive = 0;
        }

        if (p2_moves) {
            const Move& m = p2.get_active().moves[a2.index];
            if (m.effect != MoveEffect::PROTECT) p2.get_active().protect_consecutive = 0;
        } else {
            p2.get_active().protect_consecutive = 0;
        }

        if (p1_moves || p2_moves) {
            // Determine turn execution order (Priority first, then Speed)
            bool p1_goes_first = true;
            if (p1_moves && p2_moves) {
                int p1_pri = p1.get_active().moves[a1.index].priority;
                int p2_pri = p2.get_active().moves[a2.index].priority;
                
                if (p1_pri > p2_pri) {
                    p1_goes_first = true;
                } else if (p1_pri < p2_pri) {
                    p1_goes_first = false;
                } else {
                    int p1_speed = p1.get_active().get_modified_spe(get_active_weather());
                    int p2_speed = p2.get_active().get_modified_spe(get_active_weather());
                    
                    if (p1_speed < p2_speed) {
                        p1_goes_first = false;
                    } else if (p1_speed == p2_speed) {
                        p1_goes_first = (p1.name <= p2.name);
                    }
                }
            } else {
                p1_goes_first = p1_moves;
            }

            // Execution slots
            auto execute_move = [&](Side& attacker_side, Side& defender_side, int move_idx) {
                Pokemon& attacker = attacker_side.get_active();
                Pokemon& defender = defender_side.get_active();

                if (attacker.is_fainted()) return;
                if (move_idx < 0 || move_idx >= static_cast<int>(attacker.moves.size())) return;

                const Move& move = attacker.moves[move_idx];

                // Damp ability blocks explosion/KO moves
                if (move.id == "explosion" || move.id == "selfdestruct") {
                    if (attacker.ability == "Damp" || defender.ability == "Damp") return;
                }

                // Truant loaf check
                if (attacker.ability == "Truant") {
                    if (attacker.truant_turn) {
                        attacker.truant_turn = false;
                        return; // Loafing
                    } else {
                        attacker.truant_turn = true;
                    }
                }

                // Choice item move lock verification
                if (attacker.item == "Choice Band") {
                    if (attacker.locked_move_idx == -1) {
                        attacker.locked_move_idx = move_idx;
                    } else if (attacker.locked_move_idx != move_idx) {
                        return;
                    }
                }
                
                // Sleep status check with Early Bird fast sleep counter reduction
                if (attacker.status == Status::SLP) {
                    int drop = (attacker.ability == "Early Bird") ? 2 : 1;
                    attacker.sleep_turns = std::max(0, attacker.sleep_turns - drop);
                    if (attacker.sleep_turns <= 0) {
                        attacker.status = Status::NONE; // Wakes up
                    } else {
                        return; // Asleep
                    }
                }

                // Freeze status check
                if (attacker.status == Status::FRZ) {
                    std::uniform_int_distribution<int> thaw_dist(0, 4);
                    if (thaw_dist(rng) == 0) {
                        attacker.status = Status::NONE; // Thaws out
                    } else {
                        return; // Frozen
                    }
                }
                
                // Taunt prevents Status moves from executing
                if (attacker.taunt_turns > 0 && move.category == Category::STATUS) {
                    return;
                }

                // Protect blocks all moves targeting this Pokémon
                if (defender.is_protected && move.effect != MoveEffect::WEATHER_SUN && 
                    move.effect != MoveEffect::WEATHER_RAIN && move.effect != MoveEffect::WEATHER_SAND && 
                    move.effect != MoveEffect::WEATHER_HAIL) {
                    return; // Blocked by protect
                }

                // Soundproof blocks sound moves
                bool is_sound_move = (move.id == "roar" || move.id == "growl" || move.id == "sing" || move.id == "screech" || move.id == "hypervoice");
                if (is_sound_move && defender.ability == "Soundproof") {
                    return; // Blocked
                }

                // Substitute blocks status moves targeting this Pokémon
                if (defender.substitute_hp > 0 && move.category == Category::STATUS && 
                    move.effect != MoveEffect::RECOVERY && move.effect != MoveEffect::STAT_BOOST) {
                    return; // Blocks Status moves
                }

                // Accuracy check
                if (move.accuracy > 0) {
                    double acc_mult = Pokemon::get_acc_eva_multiplier(attacker.boost_acc);
                    double eva_mult = Pokemon::get_acc_eva_multiplier(defender.boost_eva);
                    double hit_chance = move.accuracy * (acc_mult / eva_mult);
                    
                    if (defender.ability == "Sand Veil" && get_active_weather() == Weather::SANDSTORM) {
                        hit_chance *= 0.8;
                    }
                    if (attacker.ability == "Hustle" && move.category == Category::PHYSICAL) {
                        hit_chance *= 0.8;
                    }
                    if (attacker.ability == "Compound Eyes") {
                        hit_chance *= 1.3;
                    }
                    
                    std::uniform_int_distribution<int> hit_dist(0, 99);
                    if (hit_dist(rng) >= hit_chance) {
                        return; // Miss
                    }
                }

                // Type Absorptions (Volt Absorb, Water Absorb, Flash Fire)
                if (move.category != Category::STATUS) {
                    if (move.type == Type::ELECTRIC && defender.ability == "Volt Absorb") {
                        int heal = defender.max_hp / 4;
                        defender.hp = std::min(defender.max_hp, defender.hp + heal);
                        return; // Immune
                    }
                    if (move.type == Type::WATER && defender.ability == "Water Absorb") {
                        int heal = defender.max_hp / 4;
                        defender.hp = std::min(defender.max_hp, defender.hp + heal);
                        return; // Immune
                    }
                    if (move.type == Type::FIRE && defender.ability == "Flash Fire") {
                        defender.flash_fire_active = true;
                        return; // Immune
                    }
                }
                
                // 1. Handle damaging move
                if (move.category != Category::STATUS) {
                    if (move.ohko && defender.ability == "Sturdy") {
                        return; // blocked by Sturdy
                    }

                    int hits = 1;
                    if (move.max_hits > 1) {
                        std::uniform_int_distribution<int> hit_cnt_dist(move.min_hits, move.max_hits);
                        hits = hit_cnt_dist(rng);
                    }

                    int total_damage_dealt = 0;
                    for (int h = 0; h < hits; ++h) {
                        if (defender.is_fainted()) break;

                        std::uniform_int_distribution<int> crit_dist(0, 15);
                        bool is_crit = (crit_dist(rng) == 0);
                        
                        int damage = 0;
                        if (move.ohko) {
                            damage = defender.hp;
                        } else {
                            damage = calculate_damage(attacker, defender, move, 1.0f, get_active_weather(), is_crit);
                        }

                        if (defender.substitute_hp > 0) {
                            defender.substitute_hp = std::max(0, defender.substitute_hp - damage);
                        } else {
                            defender.hp = std::max(0, defender.hp - damage);
                            total_damage_dealt += damage;
                            if (defender.hp == 0) {
                                defender.status = Status::FNT;
                                if (defender.destiny_bond_active) {
                                    attacker.hp = 0;
                                    attacker.status = Status::FNT;
                                }
                            }
                        }

                        // Fire move thaws frozen target
                        if (defender.status == Status::FRZ && move.type == Type::FIRE) {
                            defender.status = Status::NONE;
                        }
                    }

                    // Recoil damage
                    if (move.recoil_factor > 0.0f && total_damage_dealt > 0 && !attacker.is_fainted() && attacker.ability != "Rock Head") {
                        int recoil = std::floor(total_damage_dealt * move.recoil_factor);
                        attacker.hp = std::max(0, attacker.hp - recoil);
                        if (attacker.hp == 0) attacker.status = Status::FNT;
                    }

                    // Drain healing
                    if (move.drain_factor > 0.0f && total_damage_dealt > 0 && !attacker.is_fainted()) {
                        if (defender.ability == "Liquid Ooze") {
                            int ooze_dmg = std::floor(total_damage_dealt * move.drain_factor);
                            attacker.hp = std::max(0, attacker.hp - ooze_dmg);
                            if (attacker.hp == 0) attacker.status = Status::FNT;
                        } else {
                            int heal = std::floor(total_damage_dealt * move.drain_factor);
                            attacker.hp = std::min(attacker.max_hp, attacker.hp + heal);
                        }
                    }

                    // Self stat modification on hit (e.g. Overheat)
                    if (!move.self_boost_stat.empty() && !attacker.is_fainted()) {
                        int stage = move.self_boost_stage;
                        if (move.self_boost_stat == "atk") attacker.apply_boost(attacker.boost_atk, stage);
                        else if (move.self_boost_stat == "def") attacker.apply_boost(attacker.boost_def, stage);
                        else if (move.self_boost_stat == "spa") attacker.apply_boost(attacker.boost_spa, stage);
                        else if (move.self_boost_stat == "spd") attacker.apply_boost(attacker.boost_spd, stage); // map spd secondary field correctly
                        else if (move.self_boost_stat == "spe") attacker.apply_boost(attacker.boost_spe, stage);
                        else if (move.self_boost_stat == "acc") attacker.apply_boost(attacker.boost_acc, stage);
                        else if (move.self_boost_stat == "eva") attacker.apply_boost(attacker.boost_eva, stage);
                    }

                    // Self KO move (e.g. Explosion)
                    if (move.self_ko) {
                        attacker.hp = 0;
                        attacker.status = Status::FNT;
                    }

                    // Rapid Spin clears hazards
                    if (move.effect == MoveEffect::RAPID_SPIN) {
                        attacker_side.spikes_layers = 0;
                    }

                    // Contact Abilities & Rough Skin (only physical moves on target with no substitute)
                    if (move.category == Category::PHYSICAL && !attacker.is_fainted() && defender.substitute_hp == 0) {
                        if (defender.ability == "Rough Skin") {
                            int skin_dmg = attacker.max_hp / 16;
                            attacker.hp = std::max(0, attacker.hp - skin_dmg);
                            if (attacker.hp == 0) attacker.status = Status::FNT;
                        }
                        
                        std::uniform_int_distribution<int> contact_dist(0, 99);
                        int contact_roll = contact_dist(rng);
                        if (defender.ability == "Static" && contact_roll < 33) {
                            if (attacker.status == Status::NONE && attacker.ability != "Limber") {
                                attacker.status = Status::PAR;
                            }
                        } else if (defender.ability == "Flame Body" && contact_roll < 33) {
                            if (attacker.status == Status::NONE && attacker.type1 != Type::FIRE && attacker.type2 != Type::FIRE && attacker.ability != "Water Veil") {
                                attacker.status = Status::BRN;
                            }
                        } else if (defender.ability == "Poison Point" && contact_roll < 33) {
                            if (attacker.status == Status::NONE && attacker.type1 != Type::STEEL && attacker.type2 != Type::STEEL && attacker.ability != "Immunity") {
                                attacker.status = Status::TOX;
                                attacker.toxic_counter = 1;
                            }
                        } else if (defender.ability == "Effect Spore" && contact_roll < 10) {
                            if (attacker.status == Status::NONE) {
                                std::uniform_int_distribution<int> spore_dist(0, 2);
                                int choice = spore_dist(rng);
                                if (choice == 0 && attacker.ability != "Limber") attacker.status = Status::PAR;
                                else if (choice == 1 && attacker.type1 != Type::FIRE && attacker.type2 != Type::FIRE && attacker.ability != "Water Veil") attacker.status = Status::BRN;
                                else if (choice == 2 && attacker.type1 != Type::STEEL && attacker.type2 != Type::STEEL && attacker.ability != "Immunity") {
                                    attacker.status = Status::TOX;
                                    attacker.toxic_counter = 1;
                                }
                            }
                        }
                    }
                } 
                // 2. Handle Status moves
                else {
                    if (move.effect == MoveEffect::RECOVERY) {
                        int heal_amt = std::floor(attacker.max_hp * 0.5);
                        attacker.hp = std::min(attacker.max_hp, attacker.hp + heal_amt);
                    } else if (move.effect == MoveEffect::STAT_BOOST) {
                        attacker.apply_boost(attacker.boost_atk, move.boost_atk);
                        attacker.apply_boost(attacker.boost_def, move.boost_def);
                        attacker.apply_boost(attacker.boost_spa, move.boost_spa);
                        attacker.apply_boost(attacker.boost_spd, move.boost_spd);
                        attacker.apply_boost(attacker.boost_spe, move.boost_spe);
                    } else if (move.effect == MoveEffect::STATUS_MOVE) {
                        if (defender.status == Status::NONE && !defender.is_fainted() && can_be_statused(move.status_to_inflict, defender)) {
                            // Steel is immune to Poison status moves
                            if (move.status_to_inflict == Status::TOX && (defender.type1 == Type::STEEL || defender.type2 == Type::STEEL)) {
                                // Immune
                            } else {
                                defender.status = move.status_to_inflict;
                                if (move.status_to_inflict == Status::TOX) {
                                    defender.toxic_counter = 1;
                                } else if (move.status_to_inflict == Status::SLP) {
                                    std::uniform_int_distribution<int> slp_dist(1, 3);
                                    defender.sleep_turns = slp_dist(rng);
                                }

                                // Synchronize trigger
                                if (defender.ability == "Synchronize" && attacker.status == Status::NONE && can_be_statused(move.status_to_inflict, attacker)) {
                                    if (move.status_to_inflict == Status::TOX && (attacker.type1 == Type::STEEL || attacker.type2 == Type::STEEL)) {
                                        // Immune
                                    } else {
                                        attacker.status = move.status_to_inflict;
                                        if (move.status_to_inflict == Status::TOX) {
                                            attacker.toxic_counter = 1;
                                        }
                                    }
                                }
                            }
                        }
                    } else if (move.effect == MoveEffect::TAUNT) {
                        if (!defender.is_fainted()) {
                            defender.taunt_turns = 3;
                        }
                    } else if (move.effect == MoveEffect::HAZARDS) {
                        defender_side.spikes_layers = std::min(3, defender_side.spikes_layers + 1);
                    } else if (move.effect == MoveEffect::TRICK) {
                        // Sticky Hold prevents items swap
                        if (!defender.is_fainted() && attacker.ability != "Sticky Hold" && defender.ability != "Sticky Hold") {
                            std::swap(attacker.item, defender.item);
                        }
                    } else if (move.effect == MoveEffect::PROTECT) {
                        double success_chance = 100.0 / std::pow(2.0, attacker.protect_consecutive);
                        std::uniform_int_distribution<int> prot_dist(0, 99);
                        if (prot_dist(rng) < success_chance) {
                            attacker.is_protected = true;
                            attacker.protect_consecutive++;
                        } else {
                            attacker.protect_consecutive = 0;
                        }
                    } else if (move.effect == MoveEffect::SUBSTITUTE) {
                        int cost = attacker.max_hp / 4;
                        if (attacker.hp > cost && attacker.substitute_hp == 0) {
                            attacker.hp -= cost;
                            attacker.substitute_hp = cost;
                        }
                    } else if (move.effect == MoveEffect::LEECH_SEED) {
                        if (defender.type1 != Type::GRASS && defender.type2 != Type::GRASS && !defender.is_seeded) {
                            defender.is_seeded = true;
                        }
                    } else if (move.effect == MoveEffect::DESTINY_BOND) {
                        attacker.destiny_bond_active = true;
                    } else if (move.effect == MoveEffect::WEATHER_SUN) {
                        weather = Weather::SUN;
                        weather_turns = 5;
                        update_forecast(p1, weather);
                        update_forecast(p2, weather);
                    } else if (move.effect == MoveEffect::WEATHER_RAIN) {
                        weather = Weather::RAIN;
                        weather_turns = 5;
                        update_forecast(p1, weather);
                        update_forecast(p2, weather);
                    } else if (move.effect == MoveEffect::WEATHER_SAND) {
                        weather = Weather::SANDSTORM;
                        weather_turns = 5;
                        update_forecast(p1, weather);
                        update_forecast(p2, weather);
                    } else if (move.effect == MoveEffect::WEATHER_HAIL) {
                        weather = Weather::HAIL;
                        weather_turns = 5;
                        update_forecast(p1, weather);
                        update_forecast(p2, weather);
                    }
                }

                // 3. Handle Secondary Effects on hit (only for damaging moves)
                if (move.category != Category::STATUS && move.secondary_chance > 0 && !defender.is_fainted()) {
                    int sec_chance = move.secondary_chance;
                    // Secondary effect chance doubling with Serene Grace
                    if (attacker.ability == "Serene Grace") {
                        sec_chance *= 2;
                    }

                    std::uniform_int_distribution<int> sec_dist(0, 99);
                    if (sec_dist(rng) < sec_chance) {
                        // Apply status secondary effect (cannot affect behind substitute)
                        if (defender.substitute_hp == 0 && move.secondary_status != Status::NONE && defender.status == Status::NONE && can_be_statused(move.secondary_status, defender)) {
                            if (move.secondary_status == Status::TOX && (defender.type1 == Type::STEEL || defender.type2 == Type::STEEL)) {
                                // immune
                            } else {
                                defender.status = move.secondary_status;
                                if (move.secondary_status == Status::TOX) {
                                    defender.toxic_counter = 1;
                                } else if (move.secondary_status == Status::SLP) {
                                    std::uniform_int_distribution<int> slp_dist(1, 3);
                                    defender.sleep_turns = slp_dist(rng);
                                }
                            }
                        }
                        
                        // Apply stat modification secondary effect (Clear Body, White Smoke, Hyper Cutter, Keen Eye block drops)
                        if (!move.secondary_boost_stat.empty()) {
                            int stage = move.secondary_boost_stage;
                            Pokemon& target = (stage < 0) ? defender : attacker;
                            
                            // Drops blocked by substitute & immunities
                            if (stage > 0 || target.substitute_hp == 0) {
                                if (stage < 0) {
                                    if (target.ability == "Clear Body" || target.ability == "White Smoke") return;
                                    if (target.ability == "Hyper Cutter" && move.secondary_boost_stat == "atk") return;
                                    if (target.ability == "Keen Eye" && move.secondary_boost_stat == "acc") return;
                                }

                                if (move.secondary_boost_stat == "atk") target.apply_boost(target.boost_atk, stage);
                                else if (move.secondary_boost_stat == "def") target.apply_boost(target.boost_def, stage);
                                else if (move.secondary_boost_stat == "spa") target.apply_boost(target.boost_spa, stage);
                                else if (move.secondary_boost_stat == "spd") target.apply_boost(target.boost_spd, stage);
                                else if (move.secondary_boost_stat == "spe") target.apply_boost(target.boost_spe, stage);
                                else if (move.secondary_boost_stat == "acc") target.apply_boost(target.boost_acc, stage);
                                else if (move.secondary_boost_stat == "eva") target.apply_boost(target.boost_eva, stage);
                            }
                        }
                    }
                }
            };

            if (p1_goes_first) {
                if (p1_moves) execute_move(p1, p2, a1.index);
                if (p2_moves) execute_move(p2, p1, a2.index);
            } else {
                if (p2_moves) execute_move(p2, p1, a2.index);
                if (p1_moves) execute_move(p1, p2, a1.index);
            }
        }

        // --- STEP 4: End of Turn Phase ---
        
        // 1. Weather timer decrement
        if (weather_turns > 0) {
            weather_turns--;
            if (weather_turns == 0) {
                weather = Weather::NONE;
                update_forecast(p1, weather);
                update_forecast(p2, weather);
            }
        }

        // 2. Weather chip damage (Sand Veil immune to Sandstorm chip)
        auto apply_weather_damage = [&](Side& side) {
            Pokemon& p = side.get_active();
            if (p.is_fainted()) return;
            Weather active_w = get_active_weather();
            if (active_w == Weather::SANDSTORM) {
                if (p.ability == "Sand Veil") return;
                if (p.type1 != Type::ROCK && p.type2 != Type::ROCK &&
                    p.type1 != Type::GROUND && p.type2 != Type::GROUND &&
                    p.type1 != Type::STEEL && p.type2 != Type::STEEL) {
                    p.hp = std::max(0, p.hp - static_cast<int>(std::floor(p.max_hp / 16.0)));
                    if (p.hp == 0) p.status = Status::FNT;
                }
            } else if (active_w == Weather::HAIL) {
                if (p.type1 != Type::ICE && p.type2 != Type::ICE) {
                    p.hp = std::max(0, p.hp - static_cast<int>(std::floor(p.max_hp / 16.0)));
                    if (p.hp == 0) p.status = Status::FNT;
                }
            }
        };
        apply_weather_damage(p1);
        apply_weather_damage(p2);

        // 3. Rain Dish weather healing
        auto apply_rain_dish = [&](Side& side) {
            Pokemon& p = side.get_active();
            if (p.is_fainted()) return;
            if (p.ability == "Rain Dish" && get_active_weather() == Weather::RAIN) {
                p.hp = std::min(p.max_hp, p.hp + static_cast<int>(std::floor(p.max_hp / 16.0)));
            }
        };
        apply_rain_dish(p1);
        apply_rain_dish(p2);

        // 4. Leech Seed resolution (Liquid Ooze damages healer instead)
        auto apply_leech_seed = [](Side& seeded_side, Side& healing_side) {
            Pokemon& seeded = seeded_side.get_active();
            Pokemon& healing = healing_side.get_active();
            if (seeded.is_fainted() || !seeded.is_seeded) return;
            
            int seed_dmg = std::floor(seeded.max_hp / 8.0);
            seeded.hp = std::max(0, seeded.hp - seed_dmg);
            if (seeded.hp == 0) seeded.status = Status::FNT;
            
            if (!healing.is_fainted()) {
                if (seeded.ability == "Liquid Ooze") {
                    healing.hp = std::max(0, healing.hp - seed_dmg);
                    if (healing.hp == 0) healing.status = Status::FNT;
                } else {
                    healing.hp = std::min(healing.max_hp, healing.hp + seed_dmg);
                }
            }
        };
        apply_leech_seed(p1, p2);
        apply_leech_seed(p2, p1);

        // 5. Burn & Toxic damage
        auto apply_status_damage = [](Side& side) {
            Pokemon& p = side.get_active();
            if (p.is_fainted()) return;

            if (p.status == Status::BRN) {
                int dmg = std::floor(p.max_hp / 8.0);
                p.hp = std::max(0, p.hp - dmg);
                if (p.hp == 0) {
                    p.status = Status::FNT;
                }
            } else if (p.status == Status::TOX) {
                int dmg = std::floor(p.max_hp * p.toxic_counter / 16.0);
                p.hp = std::max(0, p.hp - dmg);
                p.toxic_counter++;
                if (p.hp == 0) {
                    p.status = Status::FNT;
                }
            }
        };
        apply_status_damage(p1);
        apply_status_damage(p2);

        // 6. Leftovers Healing
        auto apply_leftovers_healing = [](Side& side) {
            Pokemon& p = side.get_active();
            if (p.is_fainted() || p.item != "Leftovers") return;
            int heal = std::floor(p.max_hp / 16.0);
            p.hp = std::min(p.max_hp, p.hp + heal);
        };
        apply_leftovers_healing(p1);
        apply_leftovers_healing(p2);

        // 7. Speed Boost ability
        auto apply_speed_boost = [](Side& side) {
            Pokemon& p = side.get_active();
            if (p.is_fainted() || p.ability != "Speed Boost") return;
            p.apply_boost(p.boost_spe, 1);
        };
        apply_speed_boost(p1);
        apply_speed_boost(p2);

        // 8. Shed Skin status cure
        auto apply_shed_skin = [&](Side& side) {
            Pokemon& p = side.get_active();
            if (p.is_fainted() || p.status == Status::NONE) return;
            if (p.ability == "Shed Skin") {
                std::uniform_int_distribution<int> shed_dist(0, 2);
                if (shed_dist(rng) == 0) {
                    p.status = Status::NONE;
                    p.toxic_counter = 0;
                }
            }
        };
        apply_shed_skin(p1);
        apply_shed_skin(p2);

        // 9. Decrement Taunt counters
        auto decrement_taunt = [](Side& side) {
            Pokemon& p = side.get_active();
            if (p.taunt_turns > 0) {
                p.taunt_turns--;
            }
        };
        decrement_taunt(p1);
        decrement_taunt(p2);

        // 10. Clean temporary turn tags (Destiny Bond)
        p1.get_active().destiny_bond_active = false;
        p2.get_active().destiny_bond_active = false;

        // --- STEP 4b: Auto-resolve forced switches after faints ---
        // If the active Pokémon fainted (from end-of-turn effects or combat),
        // immediately switch to the first available benched Pokémon.
        // This prevents MCTS from seeing "switch-only" states on the next turn.
        auto resolve_faint = [&](Side& side) {
            if (side.get_active().is_fainted() && side.has_usable_pokemon()) {
                for (int i = 0; i < (int)side.team.size(); i++) {
                    if (i != side.active_idx && !side.team[i].is_fainted()) {
                        execute_switch(side, i);
                        break;
                    }
                }
            }
        };
        resolve_faint(p1);
        resolve_faint(p2);

        // --- STEP 5: Victory/Defeat Check ---
        bool p1_alive = p1.has_usable_pokemon();
        bool p2_alive = p2.has_usable_pokemon();

        if (!p1_alive && !p2_alive) {
            winner = "Tie";
        } else if (!p1_alive) {
            winner = p2.name;
        } else if (!p2_alive) {
            winner = p1.name;
        }

        turn_count++;
    }
};

#endif // BATTLE_H
