#include "types.h"
#include "pokemon.h"
#include "move.h"
#include "move_db.h"
#include "battle.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>

// Helper to convert string to Type
Type string_to_type(const char* s) {
    if (!s) return Type::NONE;
    std::string lower(s);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "normal") return Type::NORMAL;
    if (lower == "fire") return Type::FIRE;
    if (lower == "water") return Type::WATER;
    if (lower == "grass") return Type::GRASS;
    if (lower == "electric") return Type::ELECTRIC;
    if (lower == "ice") return Type::ICE;
    if (lower == "fighting") return Type::FIGHTING;
    if (lower == "poison") return Type::POISON;
    if (lower == "ground") return Type::GROUND;
    if (lower == "flying") return Type::FLYING;
    if (lower == "psychic") return Type::PSYCHIC;
    if (lower == "bug") return Type::BUG;
    if (lower == "rock") return Type::ROCK;
    if (lower == "ghost") return Type::GHOST;
    if (lower == "dragon") return Type::DRAGON;
    if (lower == "steel") return Type::STEEL;
    if (lower == "dark") return Type::DARK;
    return Type::NONE;
}

// State cache for cloning/branching
std::unordered_map<int, Battle*> state_cache;
int next_cache_id = 1;

// Wrapper structure to allow incremental construction before creating the Battle object
struct BattleBuilder {
    Side p1;
    Side p2;
    unsigned int seed;
};

extern "C" {

    void* create_battle_builder(const char* p1_name, const char* p2_name, unsigned int seed) {
        BattleBuilder* builder = new BattleBuilder();
        builder->p1.name = p1_name ? p1_name : "Player 1";
        builder->p2.name = p2_name ? p2_name : "Player 2";
        builder->seed = seed;
        return builder;
    }

    void add_pokemon(void* builder_ptr, int player, const char* species_id, const char* name, int level,
                     int hp, int max_hp, int atk, int def, int spa, int spd, int spe,
                     const char* type1_str, const char* type2_str, const char* ability, const char* item) {
        BattleBuilder* builder = static_cast<BattleBuilder*>(builder_ptr);
        Side& side = (player == 1) ? builder->p1 : builder->p2;

        Pokemon p;
        p.id = species_id ? species_id : "";
        p.name = name ? name : "";
        p.level = level;
        p.max_hp = max_hp;
        p.hp = hp;
        p.atk = atk;
        p.def = def;
        p.spa = spa;
        p.spd = spd;
        p.spe = spe;
        p.base_hp = max_hp; // fallback base stats to calculated for simplification
        p.base_atk = atk;
        p.base_def = def;
        p.base_spa = spa;
        p.base_spd = spd;
        p.base_spe = spe;
        p.type1 = string_to_type(type1_str);
        p.type2 = string_to_type(type2_str);
        p.ability = ability ? ability : "";
        p.item = item ? item : "";
        p.reset_volatiles();

        side.team.push_back(p);
    }

    void add_move(void* builder_ptr, int player, int pokemon_idx, const char* move_id) {
        BattleBuilder* builder = static_cast<BattleBuilder*>(builder_ptr);
        Side& side = (player == 1) ? builder->p1 : builder->p2;

        if (pokemon_idx < 0 || pokemon_idx >= static_cast<int>(side.team.size())) {
            return;
        }

        auto db = get_moves_db();
        std::string mid = move_id ? move_id : "";
        // Clean move_id to match db keys (remove hyphens, spaces, etc.)
        mid.erase(std::remove(mid.begin(), mid.end(), '-'), mid.end());
        mid.erase(std::remove(mid.begin(), mid.end(), ' '), mid.end());
        std::transform(mid.begin(), mid.end(), mid.begin(), ::tolower);

        auto it = db.find(mid);
        if (it != db.end()) {
            side.team[pokemon_idx].moves.push_back(it->second);
        } else {
            // Add a default simple physical move if not found
            side.team[pokemon_idx].moves.push_back(Move::make_physical(mid, mid, Type::NORMAL, 40));
        }
    }

    void* build_battle(void* builder_ptr) {
        BattleBuilder* builder = static_cast<BattleBuilder*>(builder_ptr);
        Battle* battle = new Battle(builder->p1, builder->p2, builder->seed);
        delete builder;
        return battle;
    }

    void run_turn(void* battle_ptr, int p1_action_type, int p1_action_idx, int p2_action_type, int p2_action_idx) {
        Battle* battle = static_cast<Battle*>(battle_ptr);

        Action a1;
        a1.type = static_cast<ActionType>(p1_action_type);
        a1.index = p1_action_idx;

        Action a2;
        a2.type = static_cast<ActionType>(p2_action_type);
        a2.index = p2_action_idx;

        battle->run_turn(a1, a2);
    }

    bool is_terminal(void* battle_ptr) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        if (battle->turn_count >= Battle::MAX_TURNS) return true;
        return !battle->p1.has_usable_pokemon() || !battle->p2.has_usable_pokemon() || !battle->winner.empty();
    }

    const char* get_winner(void* battle_ptr) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        if (!battle->winner.empty()) {
            return battle->winner.c_str();
        }
        if (!battle->p1.has_usable_pokemon() && !battle->p2.has_usable_pokemon()) {
            return "Draw";
        }
        if (!battle->p1.has_usable_pokemon()) {
            return "Player 2";
        }
        if (!battle->p2.has_usable_pokemon()) {
            return "Player 1";
        }
        return "";
    }

    int get_turn_count(void* battle_ptr) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        return battle->turn_count;
    }

    void get_pokemon_info(void* battle_ptr, int player, int pokemon_idx, int* hp, int* max_hp, char* name, char* status) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        const Side& side = (player == 1) ? battle->p1 : battle->p2;

        if (pokemon_idx < 0 || pokemon_idx >= static_cast<int>(side.team.size())) {
            *hp = 0;
            *max_hp = 0;
            strcpy(name, "");
            strcpy(status, "FNT");
            return;
        }

        const Pokemon& p = side.team[pokemon_idx];
        *hp = p.hp;
        *max_hp = p.max_hp;
        strcpy(name, p.name.c_str());
        strcpy(status, status_to_string(p.status).c_str());
    }

    int get_active_index(void* battle_ptr, int player) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        const Side& side = (player == 1) ? battle->p1 : battle->p2;
        return side.active_idx;
    }

    int get_team_size(void* battle_ptr, int player) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        const Side& side = (player == 1) ? battle->p1 : battle->p2;
        return static_cast<int>(side.team.size());
    }

    int get_valid_actions(void* battle_ptr, int player, int* action_types, int* action_indices) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        const Side& side = (player == 1) ? battle->p1 : battle->p2;
        const Side& opp = (player == 1) ? battle->p2 : battle->p1;
        const Pokemon& active = side.get_active();

        int count = 0;

        // Normal turn: can use moves or switch
        // Check if trapped
        bool trapped = battle->is_trapped(active, opp.get_active());

        // Moves
        for (int i = 0; i < static_cast<int>(active.moves.size()); ++i) {
            action_types[count] = static_cast<int>(ActionType::MOVE);
            action_indices[count] = i;
            count++;
        }

        // Switches (if not trapped)
        if (!trapped) {
            for (int i = 0; i < static_cast<int>(side.team.size()); ++i) {
                if (i != side.active_idx && !side.team[i].is_fainted()) {
                    action_types[count] = static_cast<int>(ActionType::SWITCH);
                    action_indices[count] = i;
                    count++;
                }
            }
        }

        if (count == 0) {
            action_types[count] = static_cast<int>(ActionType::PASS);
            action_indices[count] = 0;
            count++;
        }

        return count;
    }

    void delete_battle(void* battle_ptr) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        delete battle;
    }

    // --- State Cache interface for branching/cloning ---

    int cache_battle(void* battle_ptr) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        Battle* copy = new Battle(*battle);
        int id = next_cache_id++;
        state_cache[id] = copy;
        return id;
    }

    void* get_cached_battle(int id) {
        auto it = state_cache.find(id);
        if (it != state_cache.end()) {
            return it->second;
        }
        return nullptr;
    }

    void delete_cached_battle(int id) {
        auto it = state_cache.find(id);
        if (it != state_cache.end()) {
            delete it->second;
            state_cache.erase(it);
        }
    }

    void clear_cache() {
        for (auto& pair : state_cache) {
            delete pair.second;
        }
        state_cache.clear();
    }

    // --- Blazing-fast C++ Rollout implementation ---

    double run_rollout(void* battle_ptr, int player, int max_depth) {
        Battle* battle = static_cast<Battle*>(battle_ptr);
        // Create a copy of the battle to roll out on
        Battle temp(*battle);

        int depth = 0;
        int p_num = player; // 1 or 2

        while (depth < max_depth) {
            // Check if terminal
            bool p1_alive = temp.p1.has_usable_pokemon();
            bool p2_alive = temp.p2.has_usable_pokemon();
            if (!p1_alive || !p2_alive || !temp.winner.empty()) {
                break;
            }

            // Get valid actions for p1
            int p1_types[20];
            int p1_indices[20];
            int p1_count = get_valid_actions(&temp, 1, p1_types, p1_indices);

            // Get valid actions for p2
            int p2_types[20];
            int p2_indices[20];
            int p2_count = get_valid_actions(&temp, 2, p2_types, p2_indices);

            // Pick random actions
            std::uniform_int_distribution<int> p1_dist(0, p1_count - 1);
            int p1_choice = p1_dist(temp.rng);

            std::uniform_int_distribution<int> p2_dist(0, p2_count - 1);
            int p2_choice = p2_dist(temp.rng);

            Action a1;
            a1.type = static_cast<ActionType>(p1_types[p1_choice]);
            a1.index = p1_indices[p1_choice];

            Action a2;
            a2.type = static_cast<ActionType>(p2_types[p2_choice]);
            a2.index = p2_indices[p2_choice];

            temp.run_turn(a1, a2);
            depth++;
        }

        // Determine reward
        std::string winner_name = temp.winner;
        if (winner_name.empty()) {
            bool p1_alive = temp.p1.has_usable_pokemon();
            bool p2_alive = temp.p2.has_usable_pokemon();
            if (!p1_alive && !p2_alive) winner_name = "Draw";
            else if (!p1_alive) winner_name = "Player 2";
            else if (!p2_alive) winner_name = "Player 1";
        }

        if (winner_name == "Player 1") {
            return (p_num == 1) ? 1.0 : 0.0;
        } else if (winner_name == "Player 2") {
            return (p_num == 2) ? 1.0 : 0.0;
        }
        return 0.5; // Draw or unfinished
    }
}
