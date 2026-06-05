#ifndef POKEMON_H
#define POKEMON_H

#include "types.h"
#include "move.h"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

struct Pokemon {
    std::string id;
    std::string name;
    int level = 100;
    Type type1 = Type::NONE;
    Type type2 = Type::NONE;

    // Current HP
    int hp = 0;
    
    // Calculated stats
    int max_hp = 0;
    int atk = 0;
    int def = 0;
    int spa = 0;
    int spd = 0;
    int spe = 0;

    // Base stats
    int base_hp = 100;
    int base_atk = 100;
    int base_def = 100;
    int base_spa = 100;
    int base_spd = 100;
    int base_spe = 100;

    // Boost stages: ranges from -6 to +6
    int boost_atk = 0;
    int boost_def = 0;
    int boost_spa = 0;
    int boost_spd = 0;
    int boost_spe = 0;
    int boost_acc = 0;
    int boost_eva = 0;

    Status status = Status::NONE;
    std::string ability = "";
    std::string item = "";
    int taunt_turns = 0;
    int sleep_turns = 0;
    int toxic_counter = 0;
    int locked_move_idx = -1;
    bool is_protected = false;
    int protect_consecutive = 0;
    int substitute_hp = 0;
    bool is_seeded = false;
    bool destiny_bond_active = false;
    
    // Ability-specific volatiles
    bool flash_fire_active = false;
    bool truant_turn = false;

    std::vector<Move> moves;

    // Calculate stats using standard main series formula (defaults to IV=31, EV=85 like Showdown Random Battles)
    void calculate_stats(int iv = 31, int ev = 85) {
        double ev_factor = std::floor(ev / 4.0);
        
        // HP Stat formula
        max_hp = std::floor((2.0 * base_hp + iv + ev_factor) * level / 100.0) + level + 10;
        hp = max_hp;

        // General Stat formula
        auto calc_stat = [&](int base) {
            return std::floor((2.0 * base + iv + ev_factor) * level / 100.0) + 5;
        };

        atk = calc_stat(base_atk);
        def = calc_stat(base_def);
        spa = calc_stat(base_spa);
        spd = calc_stat(base_spd);
        spe = calc_stat(base_spe);
    }

    bool is_fainted() const {
        return hp <= 0 || status == Status::FNT;
    }

    void reset_boosts() {
        boost_atk = 0;
        boost_def = 0;
        boost_spa = 0;
        boost_spd = 0;
        boost_spe = 0;
        boost_acc = 0;
        boost_eva = 0;
    }

    // Resets temporary battle-only volatile variables
    void reset_volatiles() {
        reset_boosts();
        taunt_turns = 0;
        toxic_counter = 0;
        locked_move_idx = -1;
        is_protected = false;
        protect_consecutive = 0;
        substitute_hp = 0;
        is_seeded = false;
        destiny_bond_active = false;
        flash_fire_active = false;
        truant_turn = false;
    }

    void apply_boost(int& stage, int amount) {
        stage = std::clamp(stage + amount, -6, 6);
    }

    // Helper: get the multiplier for stat boost stages
    static double get_boost_multiplier(int stage) {
        if (stage >= 0) {
            return (2.0 + stage) / 2.0;
        } else {
            return 2.0 / (2.0 - stage);
        }
    }

    // Helper: get accuracy / evasion multiplier
    static double get_acc_eva_multiplier(int stage) {
        if (stage >= 0) {
            return (3.0 + stage) / 3.0;
        } else {
            return 3.0 / (3.0 - stage);
        }
    }

    // Modified stats taking boosts and item/ability into account
    int get_modified_atk() const {
        double mul = get_boost_multiplier(boost_atk);
        if (status != Status::NONE && ability == "Guts") {
            mul *= 1.5;
        }
        if (ability == "Huge Power" || ability == "Pure Power") {
            mul *= 2.0;
        }
        if (ability == "Hustle") {
            mul *= 1.5;
        }
        if (item == "Choice Band") {
            mul *= 1.5;
        }
        return std::floor(atk * mul);
    }

    int get_modified_def() const {
        double mul = get_boost_multiplier(boost_def);
        if (status != Status::NONE && ability == "Marvel Scale") {
            mul *= 1.5;
        }
        return std::floor(def * mul);
    }

    int get_modified_spa() const {
        double mul = get_boost_multiplier(boost_spa);
        return std::floor(spa * mul);
    }

    int get_modified_spd(Weather weather = Weather::NONE) const {
        double mul = get_boost_multiplier(boost_spd);
        (void)weather; 
        return std::floor(spd * mul);
    }
    
    int get_modified_spe(Weather weather = Weather::NONE) const {
        double mul = get_boost_multiplier(boost_spe);
        // Swift Swim and Chlorophyll weather boosts
        if (weather == Weather::SUN && ability == "Chlorophyll") {
            mul *= 2.0;
        } else if (weather == Weather::RAIN && ability == "Swift Swim") {
            mul *= 2.0;
        }
        
        // Paralysis cuts speed by 75% in Gen 3 (Speed becomes 25% of current value)
        if (status == Status::PAR && ability != "Limber") {
            mul *= 0.25;
        }
        return std::floor(spe * mul);
    }
};

#endif // POKEMON_H
