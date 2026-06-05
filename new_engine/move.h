#ifndef MOVE_H
#define MOVE_H

#include "types.h"
#include <string>

enum class MoveEffect {
    NONE,
    RECOVERY,     // Heals 50% max HP
    STAT_BOOST,   // Boosts self stats
    STATUS_MOVE,  // Inflicts a status condition
    TAUNT,        // Inflicts Taunt (prevents status moves)
    HAZARDS,      // Sets up spikes
    TRICK,        // Swaps held items
    PROTECT,      // Blocks moves for the turn
    SUBSTITUTE,   // Spawns substitute
    LEECH_SEED,   // Seed effect
    DESTINY_BOND, // Destiny bond effect
    RAPID_SPIN,   // Clear spikes
    WEATHER_SUN,  // Sunny weather
    WEATHER_RAIN, // Rain weather
    WEATHER_SAND, // Sandstorm weather
    WEATHER_HAIL  // Hail weather
};

struct Move {
    std::string id;
    std::string name;
    Type type = Type::NONE;
    Category category = Category::STATUS;
    int base_power = 0;
    int accuracy = 100;
    int priority = 0;
    MoveEffect effect = MoveEffect::NONE;

    // Stat boost values (applied if effect == STAT_BOOST)
    int boost_atk = 0;
    int boost_def = 0;
    int boost_spa = 0;
    int boost_spd = 0;
    int boost_spe = 0;

    // Status to inflict (applied if effect == STATUS_MOVE)
    Status status_to_inflict = Status::NONE;

    // Secondary effect chance and details (e.g. 10% chance to paralyze)
    int secondary_chance = 0;
    Status secondary_status = Status::NONE;
    int secondary_boost_stage = 0;
    std::string secondary_boost_stat = ""; // "atk", "def", "spa", "spd", "spe", "acc", "eva"

    // Self-stat change on hit (e.g. Overheat lowering SpAtk by -2 stages)
    int self_boost_stage = 0;
    std::string self_boost_stat = "";

    // Drain and Recoil factors (e.g. 0.5f for 50% HP drain, 0.33f for 33% recoil)
    float drain_factor = 0.0f;
    float recoil_factor = 0.0f;

    // Custom move properties
    bool self_ko = false;
    bool ohko = false;

    // Multi-hit range
    int min_hits = 1;
    int max_hits = 1;

    // Constructors
    Move() = default;
    Move(std::string id_, std::string name_, Type type_, Category category_, int base_power_, int accuracy_, int priority_, MoveEffect effect_, int boost_atk_, int boost_def_, int boost_spa_, int boost_spd_, int boost_spe_, Status status_to_inflict_, int secondary_chance_, Status secondary_status_, int secondary_boost_stage_, std::string secondary_boost_stat_, int self_boost_stage_, std::string self_boost_stat_, float drain_factor_, float recoil_factor_, bool self_ko_, bool ohko_, int min_hits_, int max_hits_)
        : id(id_), name(name_), type(type_), category(category_), base_power(base_power_), accuracy(accuracy_), priority(priority_), effect(effect_), boost_atk(boost_atk_), boost_def(boost_def_), boost_spa(boost_spa_), boost_spd(boost_spd_), boost_spe(boost_spe_), status_to_inflict(status_to_inflict_), secondary_chance(secondary_chance_), secondary_status(secondary_status_), secondary_boost_stage(secondary_boost_stage_), secondary_boost_stat(secondary_boost_stat_), self_boost_stage(self_boost_stage_), self_boost_stat(self_boost_stat_), drain_factor(drain_factor_), recoil_factor(recoil_factor_), self_ko(self_ko_), ohko(ohko_), min_hits(min_hits_), max_hits(max_hits_) {}

    // Factory methods for common moves to simplify testing and database setup
    static Move make_physical(std::string id, std::string name, Type type, int power, int acc = 100, int pri = 0) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::PHYSICAL;
        m.base_power = power; m.accuracy = acc; m.priority = pri;
        return m;
    }

    static Move make_special(std::string id, std::string name, Type type, int power, int acc = 100, int pri = 0) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::SPECIAL;
        m.base_power = power; m.accuracy = acc; m.priority = pri;
        return m;
    }

    static Move make_recovery(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::RECOVERY; m.accuracy = 100;
        return m;
    }

    static Move make_setup(std::string id, std::string name, Type type, int atk, int def, int spa, int spd, int spe) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::STAT_BOOST; m.accuracy = 100;
        m.boost_atk = atk; m.boost_def = def; m.boost_spa = spa; m.boost_spd = spd; m.boost_spe = spe;
        return m;
    }

    static Move make_status_inflicter(std::string id, std::string name, Type type, Status status, int acc = 100) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::STATUS_MOVE; m.status_to_inflict = status; m.accuracy = acc;
        return m;
    }

    static Move make_taunt(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::TAUNT; m.accuracy = 100;
        return m;
    }

    static Move make_hazards(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::HAZARDS; m.accuracy = 100;
        return m;
    }

    static Move make_trick(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::TRICK; m.accuracy = 100;
        return m;
    }

    static Move make_protect(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::PROTECT; m.accuracy = 100; m.priority = 4;
        return m;
    }

    static Move make_substitute(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::SUBSTITUTE; m.accuracy = 100;
        return m;
    }

    static Move make_leech_seed(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::LEECH_SEED; m.accuracy = 90;
        return m;
    }

    static Move make_destiny_bond(std::string id, std::string name, Type type) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = MoveEffect::DESTINY_BOND; m.accuracy = 100;
        return m;
    }

    static Move make_rapid_spin(std::string id, std::string name, Type type, int power) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::PHYSICAL;
        m.base_power = power; m.accuracy = 100;
        m.effect = MoveEffect::RAPID_SPIN;
        return m;
    }

    static Move make_weather(std::string id, std::string name, Type type, MoveEffect weather_effect) {
        Move m;
        m.id = id; m.name = name; m.type = type; m.category = Category::STATUS;
        m.effect = weather_effect; m.accuracy = 100;
        return m;
    }
};

#endif // MOVE_H
