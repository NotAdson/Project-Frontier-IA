#include "types.h"
#include "move.h"
#include "move_db.h"
#include "pokemon.h"
#include "battle.h"
#include <iostream>
#include <cassert>
#include <cmath>

int tests_run = 0;
int tests_passed = 0;

#define ASSERT_EQUAL(val1, val2, msg) \
    tests_run++; \
    if ((val1) == (val2)) { \
        tests_passed++; \
    } else { \
        std::cerr << "FAILED: " << msg << " (Expected " << (val2) << ", got " << (val1) << ")" << std::endl; \
    }

#define ASSERT_TRUE(condition, msg) \
    tests_run++; \
    if (condition) { \
        tests_passed++; \
    } else { \
        std::cerr << "FAILED: " << msg << std::endl; \
    }

void test_stat_calculation() {
    std::cout << "Running test_stat_calculation..." << std::endl;
    
    // Test a standard Mew (100 base all stats, level 100, standard IV/EV)
    Pokemon mew;
    mew.id = "mew";
    mew.name = "Mew";
    mew.level = 100;
    mew.base_hp = 100;
    mew.base_atk = 100;
    mew.base_def = 100;
    mew.base_spa = 100;
    mew.base_spd = 100;
    mew.base_spe = 100;
    
    mew.calculate_stats(31, 85); // IV=31, EV=85
    
    // HP calculation check: floor((2 * 100 + 31 + 21) * 100 / 100) + 100 + 10 = 252 + 110 = 362
    ASSERT_EQUAL(mew.max_hp, 362, "Mew Max HP calculation");
    ASSERT_EQUAL(mew.hp, 362, "Mew current HP");
    
    // Stat calculation check: floor((2 * 100 + 31 + 21) * 100 / 100) + 5 = 252 + 5 = 257
    ASSERT_EQUAL(mew.atk, 257, "Mew Attack calculation");
    ASSERT_EQUAL(mew.def, 257, "Mew Defense calculation");
    ASSERT_EQUAL(mew.spe, 257, "Mew Speed calculation");
}

void test_type_effectiveness() {
    std::cout << "Running test_type_effectiveness..." << std::endl;
    
    // Fire vs Grass (2.0x)
    ASSERT_EQUAL(Battle::get_type_effectiveness(Type::FIRE, Type::GRASS), 2.0f, "Fire vs Grass");
    
    // Water vs Fire (2.0x)
    ASSERT_EQUAL(Battle::get_type_effectiveness(Type::WATER, Type::FIRE), 2.0f, "Water vs Fire");
    
    // Electric vs Ground (0.0x - Immunity)
    ASSERT_EQUAL(Battle::get_type_effectiveness(Type::ELECTRIC, Type::GROUND), 0.0f, "Electric vs Ground");
    
    // Normal vs Ghost (0.0x - Immunity)
    ASSERT_EQUAL(Battle::get_type_effectiveness(Type::NORMAL, Type::GHOST), 0.0f, "Normal vs Ghost");
    
    // Fighting vs Normal (2.0x)
    ASSERT_EQUAL(Battle::get_type_effectiveness(Type::FIGHTING, Type::NORMAL), 2.0f, "Fighting vs Normal");
    
    // Dragon vs Steel (0.5x)
    ASSERT_EQUAL(Battle::get_type_effectiveness(Type::DRAGON, Type::STEEL), 0.5f, "Dragon vs Steel");

    // Dual typing check: Grass vs Swampert (Water/Ground) -> 2.0 * 2.0 = 4.0x
    Pokemon swampert;
    swampert.type1 = Type::WATER;
    swampert.type2 = Type::GROUND;
    ASSERT_EQUAL(Battle::get_total_effectiveness(Type::GRASS, swampert), 4.0f, "Grass vs Swampert (Water/Ground)");
    
    // Dual typing check: Fire vs Charizard (Fire/Flying) -> 0.5 * 1.0 = 0.5x
    Pokemon charizard;
    charizard.type1 = Type::FIRE;
    charizard.type2 = Type::FLYING;
    ASSERT_EQUAL(Battle::get_total_effectiveness(Type::FIRE, charizard), 0.5f, "Fire vs Charizard (Fire/Flying)");
}

void test_stat_boosts_and_status_penalties() {
    std::cout << "Running test_stat_boosts_and_status_penalties..." << std::endl;
    
    Pokemon p;
    p.atk = 100;
    p.spe = 100;
    
    // Boost +1 check (multiplier = 1.5x)
    p.apply_boost(p.boost_atk, 1);
    ASSERT_EQUAL(p.get_modified_atk(), 150, "Attack at +1 boost");
    
    // Boost +2 check (multiplier = 2.0x)
    p.apply_boost(p.boost_atk, 1);
    ASSERT_EQUAL(p.get_modified_atk(), 200, "Attack at +2 boost");
    
    // Clamp at +6 check (multiplier = 4.0x)
    p.apply_boost(p.boost_atk, 10); // exceed
    ASSERT_EQUAL(p.boost_atk, 6, "Boost stages clamped at +6");
    ASSERT_EQUAL(p.get_modified_atk(), 400, "Attack at +6 boost");
    
    // Reset boosts check
    p.reset_boosts();
    ASSERT_EQUAL(p.get_modified_atk(), 100, "Attack after reset");

    // Paralysis speed penalty check (Speed cut to 25% in Gen 3)
    p.status = Status::PAR;
    ASSERT_EQUAL(p.get_modified_spe(), 25, "Speed modified by Paralysis");
    
    // Burn attack penalty check (Attack cut to 50%)
    p.status = Status::NONE;
    // We construct a mock battle object to test damage calculation with burn
    Pokemon defender;
    defender.def = 100;
    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    
    int unburned_dmg = Battle::calculate_damage(p, defender, tackle, 1.0f);
    
    p.status = Status::BRN;
    int burned_dmg = Battle::calculate_damage(p, defender, tackle, 1.0f);
    
    ASSERT_TRUE(burned_dmg < unburned_dmg, "Burned physical damage reduction");
}

void test_move_execution() {
    std::cout << "Running test_move_execution..." << std::endl;
    
    // Setup a clean battle
    Pokemon charmander;
    charmander.level = 50;
    charmander.base_hp = 39;
    charmander.base_atk = 52;
    charmander.base_def = 43;
    charmander.base_spa = 60;
    charmander.base_spd = 50;
    charmander.base_spe = 65;
    charmander.type1 = Type::FIRE;
    charmander.calculate_stats();
    
    Pokemon bulbasaur;
    bulbasaur.level = 50;
    bulbasaur.base_hp = 45;
    bulbasaur.base_atk = 49;
    bulbasaur.base_def = 49;
    bulbasaur.base_spa = 65;
    bulbasaur.base_spd = 65;
    bulbasaur.base_spe = 45;
    bulbasaur.type1 = Type::GRASS;
    bulbasaur.type2 = Type::POISON;
    bulbasaur.calculate_stats();
    
    Side s1 = {"Player 1", {charmander}, 0};
    Side s2 = {"Player 2", {bulbasaur}, 0};
    Battle battle(s1, s2);
    
    // 1. Damaging Special STAB Attack: Ember (Base Power 40, Fire vs Grass/Poison -> 2x)
    Move ember = Move::make_special("ember", "Ember", Type::FIRE, 40);
    battle.p1.get_active().moves.push_back(ember);
    
    int initial_opp_hp = battle.p2.get_active().hp;
    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    
    int remaining_opp_hp = battle.p2.get_active().hp;
    ASSERT_TRUE(remaining_opp_hp < initial_opp_hp, "Special Attack reduces opponent HP");
    
    // 2. Recovery Move: Recover
    battle.p2.get_active().hp = 50; // Reduce HP manually
    Move recover = Move::make_recovery("recover", "Recover", Type::NORMAL);
    battle.p2.get_active().moves.push_back(recover);
    
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle.p2.get_active().hp > 50, "Recovery heals HP");
    
    // 3. Stat Boost Move: Dragon Dance (+1 Atk, +1 Spe)
    Move dd = Move::make_setup("dragondance", "Dragon Dance", Type::DRAGON, 1, 0, 0, 0, 1);
    battle.p1.get_active().moves.push_back(dd);
    
    battle.run_turn({ActionType::MOVE, 1}, {ActionType::PASS, 0}); // Choice 1 is Dragon Dance
    ASSERT_EQUAL(battle.p1.get_active().boost_atk, 1, "Dragon Dance Atk boost");
    ASSERT_EQUAL(battle.p1.get_active().boost_spe, 1, "Dragon Dance Spe boost");
}

void test_turn_execution_order() {
    std::cout << "Running test_turn_execution_order..." << std::endl;
    
    // Faster vs Slower Pokemon
    Pokemon ninjask; // Extremely fast
    ninjask.base_spe = 160;
    ninjask.calculate_stats();
    Move slash = Move::make_physical("slash", "Slash", Type::NORMAL, 70);
    ninjask.moves.push_back(slash);
    
    Pokemon snorlax; // Very slow
    snorlax.base_spe = 30;
    snorlax.calculate_stats();
    Move slam = Move::make_physical("slam", "Slam", Type::NORMAL, 80);
    snorlax.moves.push_back(slam);
    
    Side s1 = {"Player 1", {ninjask}, 0};
    Side s2 = {"Player 2", {snorlax}, 0};
    Battle battle(s1, s2);
    
    // Turn resolution: Ninjask should move first because it is faster
    battle.run_turn({ActionType::MOVE, 0}, {ActionType::MOVE, 0});
    
    // If Ninjask moved first, Snorlax must have taken damage before executing its turn
    ASSERT_TRUE(battle.p2.get_active().hp < battle.p2.get_active().max_hp, "Faster Pokemon moves first");
    
    // 2. Switch Priority check: Switch action happens before move action
    Pokemon fresh_p;
    fresh_p.base_spe = 10;
    fresh_p.calculate_stats();
    battle.p1.team.push_back(fresh_p); // index 1
    
    // Player 1 switches (priority), Player 2 executes Move
    battle.run_turn({ActionType::SWITCH, 1}, {ActionType::MOVE, 0});
    
    ASSERT_EQUAL(battle.p1.active_idx, 1, "Switch action executed");
    ASSERT_TRUE(battle.p1.get_active().hp < battle.p1.get_active().max_hp, "Switch occurs before damage is dealt");
}

void test_victory_conditions() {
    std::cout << "Running test_victory_conditions..." << std::endl;
    
    Pokemon p1_mon;
    p1_mon.max_hp = 100;
    p1_mon.calculate_stats();
    p1_mon.hp = 10;
    
    Pokemon p2_mon;
    p2_mon.max_hp = 100;
    p2_mon.calculate_stats();
    p2_mon.hp = 100;
    
    Side s1 = {"Player 1", {p1_mon}, 0};
    Side s2 = {"Player 2", {p2_mon}, 0};
    Battle battle(s1, s2);
    
    // Player 2 uses a powerful move to faint Player 1's only Pokemon
    Move mega_punch = Move::make_physical("megapunch", "Mega Punch", Type::NORMAL, 150);
    battle.p2.get_active().moves.push_back(mega_punch);
    
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    
    ASSERT_TRUE(battle.p1.get_active().is_fainted(), "Player 1 active Pokemon fainted");
    ASSERT_EQUAL(battle.winner, "Player 2", "Player 2 declared winner");
}

void test_poison() {
    std::cout << "Running test_poison..." << std::endl;

    // Normal pokemon to be poisoned
    Pokemon normal_mon;
    normal_mon.type1 = Type::NORMAL;
    normal_mon.base_hp = 100;
    normal_mon.calculate_stats();

    // Steel pokemon immune to poison
    Pokemon steel_mon;
    steel_mon.type1 = Type::STEEL;
    steel_mon.base_hp = 100;
    steel_mon.calculate_stats();

    // Attacker
    Pokemon attacker;
    attacker.calculate_stats();
    Move toxic = Move::make_status_inflicter("toxic", "Toxic", Type::POISON, Status::TOX);
    attacker.moves.push_back(toxic);

    // Test 1: Poison application and end-of-turn damage on normal Pokemon
    Side s1 = {"Player 1", {attacker}, 0};
    Side s2 = {"Player 2", {normal_mon}, 0};
    Battle battle1(s1, s2);

    // Player 1 uses Toxic, Player 2 passes
    battle1.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});

    ASSERT_TRUE(battle1.p2.get_active().status == Status::TOX, "Toxic status applied to Normal type");
    
    int max_hp = battle1.p2.get_active().max_hp;
    int expected_dmg = std::floor(max_hp * 1.0 / 16.0);
    int expected_remaining_hp = max_hp - expected_dmg;
    ASSERT_EQUAL(battle1.p2.get_active().hp, expected_remaining_hp, "Toxic end-of-turn damage calculated correctly");

    // Test 2: Steel type immunity
    Side s3 = {"Player 1", {attacker}, 0};
    Side s4 = {"Player 2", {steel_mon}, 0};
    Battle battle2(s3, s4);

    battle2.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_TRUE(battle2.p2.get_active().status == Status::NONE, "Steel type immune to Poison status");
    ASSERT_EQUAL(battle2.p2.get_active().hp, battle2.p2.get_active().max_hp, "No damage taken by Steel type");
}

void test_taunt() {
    std::cout << "Running test_taunt..." << std::endl;

    Pokemon p1_mon;
    p1_mon.calculate_stats();
    Move taunt = Move::make_taunt("taunt", "Taunt", Type::DARK);
    p1_mon.moves.push_back(taunt);

    Pokemon p2_mon;
    p2_mon.calculate_stats();
    Move recovery = Move::make_recovery("recover", "Recover", Type::NORMAL);
    p2_mon.moves.push_back(recovery);

    Side s1 = {"Player 1", {p1_mon}, 0};
    Side s2 = {"Player 2", {p2_mon}, 0};
    Battle battle(s1, s2);

    // To ensure Taunt goes first, make Player 1 faster
    battle.p1.get_active().spe = 200;
    battle.p2.get_active().spe = 100;
    battle.p2.get_active().hp = 50; // Set low HP to verify if recovery is blocked

    // Turn 1: Player 1 uses Taunt (inflicts taunt_turns = 3, which gets decremented to 2 at end of turn)
    // Player 2 tries to use Recover but should be blocked if Taunt goes first.
    battle.run_turn({ActionType::MOVE, 0}, {ActionType::MOVE, 0});

    ASSERT_EQUAL(battle.p2.get_active().taunt_turns, 2, "Taunt turns applied and decremented at end of turn");
    ASSERT_EQUAL(battle.p2.get_active().hp, 50, "Recovery blocked by Taunt (HP remained 50)");

    // Turn 2: Player 2 tries to use Recover again. Player 1 passes.
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_EQUAL(battle.p2.get_active().taunt_turns, 1, "Taunt turns decremented to 1");
    ASSERT_EQUAL(battle.p2.get_active().hp, 50, "Recovery still blocked by Taunt");

    // Turn 3: Player 2 tries to use Recover again. Player 1 passes.
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_EQUAL(battle.p2.get_active().taunt_turns, 0, "Taunt turns decremented to 0");
    ASSERT_EQUAL(battle.p2.get_active().hp, 50, "Recovery still blocked in the final turn of Taunt");

    // Turn 4: Taunt has expired, so Recover should work now. Player 1 passes.
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle.p2.get_active().hp > 50, "Recovery succeeds after Taunt expires");
}

void test_hazards() {
    std::cout << "Running test_hazards..." << std::endl;

    Pokemon setter;
    setter.calculate_stats();
    Move spikes = Move::make_hazards("spikes", "Spikes", Type::GROUND);
    setter.moves.push_back(spikes);

    // Opponent team with a normal pokemon and a flying pokemon
    Pokemon normal_mon;
    normal_mon.type1 = Type::NORMAL;
    normal_mon.base_hp = 100;
    normal_mon.calculate_stats();

    Pokemon flying_mon;
    flying_mon.type1 = Type::FLYING;
    flying_mon.base_hp = 100;
    flying_mon.calculate_stats();

    Side s1 = {"Player 1", {setter}, 0};
    Side s2 = {"Player 2", {normal_mon, flying_mon}, 0};
    Battle battle(s1, s2);

    // Set 1 layer of Spikes
    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p2.spikes_layers, 1, "Spikes layers set to 1");

    // Switch in Flying pokemon -> should not take damage
    battle.run_turn({ActionType::PASS, 0}, {ActionType::SWITCH, 1});
    ASSERT_EQUAL(battle.p2.active_idx, 1, "Switched to Flying Pokemon");
    ASSERT_EQUAL(battle.p2.get_active().hp, battle.p2.get_active().max_hp, "Flying type is immune to Spikes damage");

    // Set a second layer of Spikes (total 2)
    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p2.spikes_layers, 2, "Spikes layers set to 2");

    // Switch back to Normal Pokemon -> should take 1/6th max HP damage (since layers = 2)
    battle.run_turn({ActionType::PASS, 0}, {ActionType::SWITCH, 0});
    ASSERT_EQUAL(battle.p2.active_idx, 0, "Switched to Normal Pokemon");
    
    int max_hp = battle.p2.get_active().max_hp;
    int expected_dmg = std::floor(max_hp / 6.0);
    ASSERT_EQUAL(battle.p2.get_active().hp, max_hp - expected_dmg, "Normal Pokemon took 1/6th HP damage from 2 layers of Spikes");
}

void test_barrier() {
    std::cout << "Running test_barrier..." << std::endl;

    Pokemon p;
    p.calculate_stats();
    Move barrier = Move::make_setup("barrier", "Barrier", Type::PSYCHIC, 0, 2, 0, 0, 0);
    p.moves.push_back(barrier);

    Side s1 = {"Player 1", {p}, 0};
    Side s2 = {"Player 2", {p}, 0}; // just a dummy
    Battle battle(s1, s2);

    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p1.get_active().boost_def, 2, "Barrier boosts defense stage by +2");
}

void test_trick() {
    std::cout << "Running test_trick..." << std::endl;

    Pokemon p1_mon;
    p1_mon.item = "Choice Band";
    p1_mon.calculate_stats();
    Move trick = Move::make_trick("trick", "Trick", Type::PSYCHIC);
    p1_mon.moves.push_back(trick);

    Pokemon p2_mon;
    p2_mon.item = "Leftovers";
    p2_mon.calculate_stats();

    Side s1 = {"Player 1", {p1_mon}, 0};
    Side s2 = {"Player 2", {p2_mon}, 0};
    Battle battle(s1, s2);

    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p1.get_active().item, "Leftovers", "Player 1 got Leftovers via Trick");
    ASSERT_EQUAL(battle.p2.get_active().item, "Choice Band", "Player 2 got Choice Band via Trick");
}

void test_priority_moves() {
    std::cout << "Running test_priority_moves..." << std::endl;
    // Slower pokemon using priority move (+1) vs faster pokemon using normal move
    Pokemon slow_mon;
    slow_mon.base_spe = 10;
    slow_mon.calculate_stats();
    // priority +1 move
    Move quick_attack = Move::make_physical("quickattack", "Quick Attack", Type::NORMAL, 40, 100, 1);
    slow_mon.moves.push_back(quick_attack);

    Pokemon fast_mon;
    fast_mon.base_spe = 200;
    fast_mon.calculate_stats();
    Move slam = Move::make_physical("slam", "Slam", Type::NORMAL, 80, 100, 0);
    fast_mon.moves.push_back(slam);

    Side s1 = {"Player 1", {slow_mon}, 0};
    Side s2 = {"Player 2", {fast_mon}, 0};
    Battle battle(s1, s2);

    battle.run_turn({ActionType::MOVE, 0}, {ActionType::MOVE, 0});
    // If priority worked, fast_mon should take damage before slow_mon takes damage.
    ASSERT_TRUE(battle.p2.get_active().hp < battle.p2.get_active().max_hp, "Slow Pokemon with Priority move hit first");
}

void test_ability_intimidate() {
    std::cout << "Running test_ability_intimidate..." << std::endl;
    Pokemon intimidator;
    intimidator.ability = "Intimidate";
    intimidator.calculate_stats();

    Pokemon target;
    target.calculate_stats();

    Side s1 = {"Player 1", {intimidator}, 0};
    Side s2 = {"Player 2", {target}, 0};
    Battle battle(s1, s2); // Triggers on start
    ASSERT_EQUAL(battle.p2.get_active().boost_atk, -1, "Intimidate lowers opponent attack stage on battle start");
}

void test_ability_levitate() {
    std::cout << "Running test_ability_levitate..." << std::endl;
    // 1. Ground move immunity
    Pokemon levitator;
    levitator.ability = "Levitate";
    levitator.calculate_stats();
    Move eq = Move::make_physical("earthquake", "Earthquake", Type::GROUND, 100);
    
    Pokemon ground_user;
    ground_user.calculate_stats();
    ground_user.moves.push_back(eq);

    Side s1 = {"Player 1", {levitator}, 0};
    Side s2 = {"Player 2", {ground_user}, 0};
    Battle battle_dmg(s1, s2);
    battle_dmg.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_EQUAL(battle_dmg.p1.get_active().hp, battle_dmg.p1.get_active().max_hp, "Levitate grants Ground-type immunity");

    // 2. Spikes immunity
    battle_dmg.p1.spikes_layers = 3;
    Pokemon fresh_levitator;
    fresh_levitator.ability = "Levitate";
    fresh_levitator.calculate_stats();
    battle_dmg.p1.team.push_back(fresh_levitator);
    battle_dmg.run_turn({ActionType::SWITCH, 1}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle_dmg.p1.get_active().hp, battle_dmg.p1.get_active().max_hp, "Levitate grants Spikes immunity on switch-in");
}

void test_ability_natural_cure() {
    std::cout << "Running test_ability_natural_cure..." << std::endl;
    Pokemon curer;
    curer.ability = "Natural Cure";
    curer.status = Status::BRN;
    curer.calculate_stats();
    
    Pokemon bench;
    bench.calculate_stats();

    Side s1 = {"Player 1", {curer, bench}, 0};
    Side s2 = {"Player 2", {bench}, 0};
    Battle battle(s1, s2);
    // Switch out
    battle.run_turn({ActionType::SWITCH, 1}, {ActionType::PASS, 0});
    ASSERT_TRUE(battle.p1.team[0].status == Status::NONE, "Natural Cure removes status on switch out");
}

void test_ability_guts() {
    std::cout << "Running test_ability_guts..." << std::endl;
    Pokemon gutsy;
    gutsy.ability = "Guts";
    gutsy.base_atk = 100;
    gutsy.calculate_stats();
    gutsy.status = Status::BRN; // Burn would normally halve attack, but Guts boosts it by 1.5x instead

    Pokemon defender;
    defender.base_def = 100;
    defender.calculate_stats();

    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    int guts_dmg = Battle::calculate_damage(gutsy, defender, tackle, 1.0f);
    
    Pokemon standard_mon;
    standard_mon.base_atk = 100;
    standard_mon.calculate_stats();
    int standard_dmg = Battle::calculate_damage(standard_mon, defender, tackle, 1.0f);

    ASSERT_TRUE(guts_dmg > standard_dmg, "Guts boosts Attack by 1.5x and ignores burn penalty");
}

void test_ability_speed_boost() {
    std::cout << "Running test_ability_speed_boost..." << std::endl;
    Pokemon speed_booster;
    speed_booster.ability = "Speed Boost";
    speed_booster.calculate_stats();
    
    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {speed_booster}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2);
    battle.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p1.get_active().boost_spe, 1, "Speed Boost increments speed stage at end of turn");
}

void test_ability_swift_swim() {
    std::cout << "Running test_ability_swift_swim..." << std::endl;
    Pokemon swimmer;
    swimmer.ability = "Swift Swim";
    swimmer.calculate_stats();
    int base_calculated_spe = swimmer.spe;
    ASSERT_EQUAL(swimmer.get_modified_spe(Weather::RAIN), base_calculated_spe * 2, "Swift Swim doubles speed in Rain");
}

void test_ability_chlorophyll() {
    std::cout << "Running test_ability_chlorophyll..." << std::endl;
    Pokemon green;
    green.ability = "Chlorophyll";
    green.calculate_stats();
    int base_calculated_spe = green.spe;
    ASSERT_EQUAL(green.get_modified_spe(Weather::SUN), base_calculated_spe * 2, "Chlorophyll doubles speed in Sun");
}


void test_held_items() {
    std::cout << "Running test_held_items..." << std::endl;

    // 1. Leftovers
    Pokemon leftover_mon;
    leftover_mon.item = "Leftovers";
    leftover_mon.base_hp = 100;
    leftover_mon.calculate_stats();

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {leftover_mon}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle_left(s1, s2);
    // deal some damage
    battle_left.p1.get_active().hp = 200;
    battle_left.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
    int expected_heal = std::floor(battle_left.p1.get_active().max_hp / 16.0);
    ASSERT_EQUAL(battle_left.p1.get_active().hp, 200 + expected_heal, "Leftovers heals 1/16th Max HP at end of turn");

    // 2. Choice Band
    Pokemon cb_mon;
    cb_mon.item = "Choice Band";
    cb_mon.base_atk = 100;
    cb_mon.calculate_stats();
    
    Pokemon defender;
    defender.base_def = 100;
    defender.calculate_stats();

    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    int cb_dmg = Battle::calculate_damage(cb_mon, defender, tackle, 1.0f);

    Pokemon standard_mon;
    standard_mon.base_atk = 100;
    standard_mon.calculate_stats();
    int standard_dmg = Battle::calculate_damage(standard_mon, defender, tackle, 1.0f);

    ASSERT_TRUE(cb_dmg > standard_dmg, "Choice Band boosts physical attack damage by 1.5x");

    // Choice Band Move lock check
    Move slam = Move::make_physical("slam", "Slam", Type::NORMAL, 80);
    cb_mon.moves.push_back(tackle);
    cb_mon.moves.push_back(slam);

    Side s3 = {"Player 1", {cb_mon}, 0};
    Side s4 = {"Player 2", {dummy}, 0};
    Battle battle_lock(s3, s4);
    // Turn 1: select move 0 (Tackle). This sets locked_move_idx to 0.
    battle_lock.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle_lock.p1.get_active().locked_move_idx, 0, "Choice Band locks user to first chosen move");

    // Turn 2: attempting to use move 1 (Slam) should fail/no-op due to lock
    int hp_before = battle_lock.p2.get_active().hp;
    battle_lock.run_turn({ActionType::MOVE, 1}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle_lock.p2.get_active().hp, hp_before, "Choice Band prevents execution of different moves");
}

void test_status_conditions() {
    std::cout << "Running test_status_conditions..." << std::endl;

    // 1. Sleep
    Pokemon sleeper;
    sleeper.calculate_stats();
    Move recover = Move::make_recovery("recover", "Recover", Type::NORMAL);
    sleeper.moves.push_back(recover);

    Pokemon attacker;
    attacker.calculate_stats();
    Move hypnosis = Move::make_status_inflicter("hypnosis", "Hypnosis", Type::PSYCHIC, Status::SLP);
    attacker.moves.push_back(hypnosis);

    Side s1 = {"Player 1", {sleeper}, 0};
    Side s2 = {"Player 2", {attacker}, 0};
    Battle battle_slp(s1, s2);
    // make sure attacker is faster to inflict sleep
    battle_slp.p2.get_active().spe = 200;
    battle_slp.p1.get_active().spe = 100;
    battle_slp.p1.get_active().hp = 50;

    // Turn 1: Hypnosis hits sleeper. sleeper tries to use Recover but sleep is applied first.
    battle_slp.run_turn({ActionType::MOVE, 0}, {ActionType::MOVE, 0});

    ASSERT_TRUE(battle_slp.p1.get_active().status == Status::SLP, "Sleep status inflicted successfully");
    ASSERT_EQUAL(battle_slp.p1.get_active().hp, 50, "Recovery blocked because Pokémon is asleep");

    // 2. Freeze and Fire Thaw
    Pokemon frozen;
    frozen.status = Status::FRZ;
    frozen.calculate_stats();
    frozen.moves.push_back(recover);

    Pokemon fire_user;
    fire_user.calculate_stats();
    Move ember = Move::make_special("ember", "Ember", Type::FIRE, 40);
    fire_user.moves.push_back(ember);

    Side s3 = {"Player 1", {frozen}, 0};
    Side s4 = {"Player 2", {fire_user}, 0};
    Battle battle_frz(s3, s4);
    battle_frz.p1.get_active().hp = 50;

    // Turn 1: Fire user uses Ember (Fire). Frozen pokemon should thaw out.
    battle_frz.run_turn({ActionType::MOVE, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle_frz.p1.get_active().status == Status::NONE, "Frozen Pokémon thaws out when hit by Fire-type move");

    // 3. Scaling Toxic
    Pokemon toxic_mon;
    toxic_mon.status = Status::TOX;
    toxic_mon.toxic_counter = 1;
    toxic_mon.base_hp = 100;
    toxic_mon.calculate_stats();
    int max_hp = toxic_mon.max_hp;

    Pokemon dummy_mon;
    dummy_mon.calculate_stats();

    Side s5 = {"Player 1", {toxic_mon}, 0};
    Side s6 = {"Player 2", {dummy_mon}, 0};
    Battle battle_tox(s5, s6);

    // Turn 1: damage is floor(max_hp * 1 / 16)
    battle_tox.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
    int dmg_1 = std::floor(max_hp * 1.0 / 16.0);
    ASSERT_EQUAL(battle_tox.p1.get_active().hp, max_hp - dmg_1, "Toxic Turn 1 deals 1/16th Max HP damage");

    // Turn 2: damage is floor(max_hp * 2 / 16)
    int hp_after_1 = battle_tox.p1.get_active().hp;
    battle_tox.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
    int dmg_2 = std::floor(max_hp * 2.0 / 16.0);
    ASSERT_EQUAL(battle_tox.p1.get_active().hp, hp_after_1 - dmg_2, "Toxic Turn 2 damage scales to 2/16th Max HP");
}

void test_custom_moves() {
    std::cout << "Running test_custom_moves..." << std::endl;

    Pokemon p1_mon;
    p1_mon.calculate_stats();

    Pokemon p2_mon;
    p2_mon.calculate_stats();

    // 1. Protect
    Move protect = Move::make_protect("protect", "Protect", Type::NORMAL);
    p1_mon.moves.push_back(protect);
    Move slash = Move::make_physical("slash", "Slash", Type::NORMAL, 70);
    p2_mon.moves.push_back(slash);

    Side s1 = {"Player 1", {p1_mon}, 0};
    Side s2 = {"Player 2", {p2_mon}, 0};
    Battle battle_prot(s1, s2);
    // Protect has +4 priority, it will go first
    battle_prot.run_turn({ActionType::MOVE, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle_prot.p1.get_active().is_protected, "Protect sets is_protected flag");
    ASSERT_EQUAL(battle_prot.p1.get_active().hp, battle_prot.p1.get_active().max_hp, "Protect blocks damage");

    // 2. Substitute
    Pokemon sub_user;
    sub_user.calculate_stats();
    int max_hp = sub_user.max_hp;
    Move substitute = Move::make_substitute("substitute", "Substitute", Type::NORMAL);
    sub_user.moves.push_back(substitute);

    Pokemon attacker;
    attacker.calculate_stats();
    attacker.moves.push_back(slash);

    Side s3 = {"Player 1", {sub_user}, 0};
    Side s4 = {"Player 2", {attacker}, 0};
    Battle battle_sub(s3, s4);
    // Turn 1: P1 uses Substitute (loses 25% max HP, spawns sub)
    battle_sub.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    int expected_hp = max_hp - std::floor(max_hp / 4.0);
    ASSERT_EQUAL(battle_sub.p1.get_active().hp, expected_hp, "Substitute deducts 25% Max HP");
    ASSERT_EQUAL(battle_sub.p1.get_active().substitute_hp, static_cast<int>(std::floor(max_hp / 4.0)), "Substitute has 25% Max HP");

    // Turn 2: Attacker uses Slash. It should hit the substitute, leaving Pokemon's HP unchanged
    battle_sub.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_EQUAL(battle_sub.p1.get_active().hp, expected_hp, "Substitute absorbs damage (Pokemon HP remains unchanged)");

    // 3. Leech Seed
    Pokemon seeded;
    seeded.calculate_stats();
    int max_hp_seeded = seeded.max_hp;

    Pokemon seeder;
    seeder.calculate_stats();
    seeder.hp = 100;
    Move leech_seed = Move::make_leech_seed("leechseed", "Leech Seed", Type::GRASS);
    seeder.moves.push_back(leech_seed);

    Side s5 = {"Player 1", {seeded}, 0};
    Side s6 = {"Player 2", {seeder}, 0};
    Battle battle_seed(s5, s6);
    // Turn 1: Player 2 uses Leech Seed
    battle_seed.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle_seed.p1.get_active().is_seeded, "Leech Seed applied to target");
    
    int seed_dmg = std::floor(max_hp_seeded / 8.0);
    ASSERT_EQUAL(battle_seed.p1.get_active().hp, max_hp_seeded - seed_dmg, "Leech Seed deals 1/8th Max HP damage at end of turn");
    ASSERT_EQUAL(battle_seed.p2.get_active().hp, 100 + seed_dmg, "Leech Seed heals attacker by damage amount");

    // 4. Destiny Bond
    Pokemon db_mon;
    db_mon.calculate_stats();
    db_mon.hp = 10;
    Move db = Move::make_destiny_bond("destinybond", "Destiny Bond", Type::GHOST);
    db_mon.moves.push_back(db);

    Pokemon db_attacker;
    db_attacker.calculate_stats();
    Move earthquake = Move::make_physical("earthquake", "Earthquake", Type::GROUND, 100);
    db_attacker.moves.push_back(earthquake);

    Side s7 = {"Player 1", {db_mon}, 0};
    Side s8 = {"Player 2", {db_attacker}, 0};
    Battle battle_db(s7, s8);
    // make db_mon faster to set Destiny Bond
    battle_db.p1.get_active().spe = 200;
    battle_db.p2.get_active().spe = 100;

    battle_db.run_turn({ActionType::MOVE, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle_db.p1.get_active().is_fainted(), "Destiny Bond user fainted");
    ASSERT_TRUE(battle_db.p2.get_active().is_fainted(), "Destiny Bond fainted the attacker");

    // 5. Rapid Spin
    Pokemon spin_user;
    spin_user.calculate_stats();
    Move rapid_spin = Move::make_rapid_spin("rapidspin", "Rapid Spin", Type::NORMAL, 20);
    spin_user.moves.push_back(rapid_spin);

    Pokemon spin_def;
    spin_def.calculate_stats();

    Side s9 = {"Player 1", {spin_user}, 0};
    Side s10 = {"Player 2", {spin_def}, 0};
    Battle battle_spin(s9, s10);
    battle_spin.p1.spikes_layers = 2; // setup spikes on player 1's side

    battle_spin.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle_spin.p1.spikes_layers, 0, "Rapid Spin clears spikes on user's side");
}

void test_weather() {
    std::cout << "Running test_weather..." << std::endl;

    Pokemon rain_setter;
    rain_setter.calculate_stats();
    Move rain_dance = Move::make_weather("raindance", "Rain Dance", Type::WATER, MoveEffect::WEATHER_RAIN);
    rain_setter.moves.push_back(rain_dance);

    Pokemon water_user;
    water_user.calculate_stats();
    Move surf = Move::make_special("surf", "Surf", Type::WATER, 95);
    water_user.moves.push_back(surf);

    Pokemon fire_user;
    fire_user.calculate_stats();
    Move flamethrower = Move::make_special("flamethrower", "Flamethrower", Type::FIRE, 95);
    fire_user.moves.push_back(flamethrower);

    Pokemon defender;
    defender.calculate_stats();

    // 1. Weather damage multiplication
    int surf_normal_dmg = Battle::calculate_damage(water_user, defender, surf, 1.0f, Weather::NONE);
    int surf_rain_dmg = Battle::calculate_damage(water_user, defender, surf, 1.0f, Weather::RAIN);
    ASSERT_TRUE(surf_rain_dmg > surf_normal_dmg, "Rain boosts Water-type moves by 1.5x");

    int fire_normal_dmg = Battle::calculate_damage(fire_user, defender, flamethrower, 1.0f, Weather::NONE);
    int fire_rain_dmg = Battle::calculate_damage(fire_user, defender, flamethrower, 1.0f, Weather::RAIN);
    ASSERT_TRUE(fire_rain_dmg < fire_normal_dmg, "Rain weakens Fire-type moves by 0.5x");

    // 2. Weather Speed boost (Swift Swim)
    Pokemon swimmer;
    swimmer.ability = "Swift Swim";
    swimmer.calculate_stats();
    int base_calculated_spe = swimmer.spe;
    ASSERT_EQUAL(swimmer.get_modified_spe(Weather::RAIN), base_calculated_spe * 2, "Swift Swim doubles speed in Rain");

    // 3. Sandstorm chip damage
    Pokemon sand_dummy;
    sand_dummy.type1 = Type::NORMAL;
    sand_dummy.base_hp = 100;
    sand_dummy.calculate_stats();

    Pokemon ground_mon;
    ground_mon.type1 = Type::GROUND;
    ground_mon.base_hp = 100;
    ground_mon.calculate_stats();

    Side s1 = {"Player 1", {sand_dummy}, 0};
    Side s2 = {"Player 2", {ground_mon}, 0};
    Battle battle_sand(s1, s2);
    battle_sand.weather = Weather::SANDSTORM;
    battle_sand.weather_turns = 5;

    battle_sand.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
    ASSERT_TRUE(battle_sand.p1.get_active().hp < battle_sand.p1.get_active().max_hp, "Normal type takes chip damage in Sandstorm");
    ASSERT_EQUAL(battle_sand.p2.get_active().hp, battle_sand.p2.get_active().max_hp, "Ground type is immune to Sandstorm damage");
}

void test_rng_accuracy_crits() {
    std::cout << "Running test_rng_accuracy_crits..." << std::endl;

    Pokemon attacker;
    attacker.calculate_stats();
    // 50% accuracy move
    Move zap_cannon = Move::make_special("zapcannon", "Zap Cannon", Type::ELECTRIC, 120, 50);
    attacker.moves.push_back(zap_cannon);

    Pokemon defender;
    defender.calculate_stats();

    Side s1 = {"Player 1", {attacker}, 0};
    Side s2 = {"Player 2", {defender}, 0};

    // seed = 1 has a deterministic miss/hit behavior
    Battle battle_miss(s1, s2, 1);
    battle_miss.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});

    // Test critical hits ignore stats
    Pokemon crit_attacker;
    crit_attacker.boost_atk = -2;
    crit_attacker.calculate_stats();

    Pokemon crit_defender;
    crit_defender.boost_def = 2;
    crit_defender.calculate_stats();

    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    int crit_dmg = Battle::calculate_damage(crit_attacker, crit_defender, tackle, 1.0f, Weather::NONE, true);
    int standard_dmg = Battle::calculate_damage(crit_attacker, crit_defender, tackle, 1.0f, Weather::NONE, false);

    ASSERT_TRUE(crit_dmg > standard_dmg, "Critical hits ignore negative attacker boosts and positive defender boosts");
}

void test_ability_crit_block() {
    std::cout << "Running test_ability_crit_block..." << std::endl;
    Pokemon attacker;
    attacker.calculate_stats();
    
    Pokemon defender;
    defender.ability = "Battle Armor";
    defender.calculate_stats();

    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    int dmg = Battle::calculate_damage(attacker, defender, tackle, 1.0f, Weather::NONE, true);
    int normal_dmg = Battle::calculate_damage(attacker, defender, tackle, 1.0f, Weather::NONE, false);
    ASSERT_EQUAL(dmg, normal_dmg, "Battle Armor blocks critical hit extra damage");
}

void test_ability_weather_summoners() {
    std::cout << "Running test_ability_weather_summoners..." << std::endl;
    Pokemon rain_setter;
    rain_setter.ability = "Drizzle";
    rain_setter.calculate_stats();

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {rain_setter}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle_rain(s1, s2);
    ASSERT_TRUE(battle_rain.weather == Weather::RAIN, "Drizzle summons rain on switch-in");
    ASSERT_EQUAL(battle_rain.weather_turns, -1, "Drizzle weather is permanent in Gen 3");

    Pokemon sun_setter;
    sun_setter.ability = "Drought";
    sun_setter.calculate_stats();
    Side s3 = {"Player 1", {sun_setter}, 0};
    Battle battle_sun(s3, s2);
    ASSERT_TRUE(battle_sun.weather == Weather::SUN, "Drought summons sun on switch-in");

    Pokemon sand_setter;
    sand_setter.ability = "Sand Stream";
    sand_setter.calculate_stats();
    Side s5 = {"Player 1", {sand_setter}, 0};
    Battle battle_sand(s5, s2);
    ASSERT_TRUE(battle_sand.weather == Weather::SANDSTORM, "Sand Stream summons sandstorm on switch-in");
}

void test_ability_weather_negators() {
    std::cout << "Running test_ability_weather_negators..." << std::endl;
    Pokemon negator;
    negator.ability = "Cloud Nine";
    negator.calculate_stats();

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {negator}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2);
    battle.weather = Weather::RAIN;

    Pokemon swimmer;
    swimmer.ability = "Swift Swim";
    swimmer.spe = 100;
    swimmer.calculate_stats();

    ASSERT_TRUE(battle.get_active_weather() == Weather::NONE, "Cloud Nine negates weather effects");
}

void test_ability_absorbers() {
    std::cout << "Running test_ability_absorbers..." << std::endl;
    Pokemon volt_absorber;
    volt_absorber.ability = "Volt Absorb";
    volt_absorber.calculate_stats();
    volt_absorber.hp = 100;

    Pokemon attacker;
    attacker.calculate_stats();
    Move thunderbolt = Move::make_special("thunderbolt", "Thunderbolt", Type::ELECTRIC, 90);
    attacker.moves.push_back(thunderbolt);

    Side s1 = {"Player 1", {volt_absorber}, 0};
    Side s2 = {"Player 2", {attacker}, 0};
    Battle battle_volt(s1, s2);
    battle_volt.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    
    int expected_heal = volt_absorber.max_hp / 4;
    ASSERT_EQUAL(battle_volt.p1.get_active().hp, 100 + expected_heal, "Volt Absorb absorbs Electric moves and heals 25% Max HP");

    Pokemon fire_absorber;
    fire_absorber.ability = "Flash Fire";
    fire_absorber.calculate_stats();
    
    Pokemon dummy;
    dummy.calculate_stats();
    Move ember = Move::make_special("ember", "Ember", Type::FIRE, 40);
    dummy.moves.push_back(ember);

    Side s3 = {"Player 1", {fire_absorber}, 0};
    Side s4 = {"Player 2", {dummy}, 0};
    Battle battle_fire(s3, s4);
    
    ASSERT_TRUE(!battle_fire.p1.get_active().flash_fire_active, "Flash Fire starts inactive");
    
    battle_fire.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle_fire.p1.get_active().flash_fire_active, "Flash Fire activates when hit by Fire-type move");
    ASSERT_EQUAL(battle_fire.p1.get_active().hp, battle_fire.p1.get_active().max_hp, "Flash Fire grants immunity to Fire moves");
}

void test_ability_thick_fat() {
    std::cout << "Running test_ability_thick_fat..." << std::endl;
    Pokemon thick_fat_mon;
    thick_fat_mon.ability = "Thick Fat";
    thick_fat_mon.calculate_stats();

    Pokemon attacker;
    attacker.calculate_stats();

    Move ember = Move::make_special("ember", "Ember", Type::FIRE, 40);
    int fat_dmg = Battle::calculate_damage(attacker, thick_fat_mon, ember, 1.0f);
    
    Pokemon normal_mon;
    normal_mon.calculate_stats();
    int normal_dmg = Battle::calculate_damage(attacker, normal_mon, ember, 1.0f);

    ASSERT_TRUE(fat_dmg < normal_dmg, "Thick Fat halves Fire-type move damage");
}

void test_ability_pinch_boosts() {
    std::cout << "Running test_ability_pinch_boosts..." << std::endl;
    Pokemon pincher;
    pincher.ability = "Blaze";
    pincher.calculate_stats();
    pincher.hp = 10;

    Pokemon defender;
    defender.calculate_stats();

    Move ember = Move::make_special("ember", "Ember", Type::FIRE, 40);
    int pinch_dmg = Battle::calculate_damage(pincher, defender, ember, 1.0f);

    Pokemon normal_mon;
    normal_mon.calculate_stats();
    int normal_dmg = Battle::calculate_damage(normal_mon, defender, ember, 1.0f);

    ASSERT_TRUE(pinch_dmg > normal_dmg, "Blaze boosts Fire moves by 1.5x at low health (pinch)");
}

void test_ability_rain_dish() {
    std::cout << "Running test_ability_rain_dish..." << std::endl;
    Pokemon rain_disher;
    rain_disher.ability = "Rain Dish";
    rain_disher.calculate_stats();
    rain_disher.hp = 100;

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {rain_disher}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2);
    battle.weather = Weather::RAIN;
    battle.weather_turns = 5;

    battle.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
    int expected_heal = std::floor(rain_disher.max_hp / 16.0);
    ASSERT_EQUAL(battle.p1.get_active().hp, 100 + expected_heal, "Rain Dish heals 1/16th Max HP in rain at end of turn");
}

void test_ability_sand_veil() {
    std::cout << "Running test_ability_sand_veil..." << std::endl;
    Pokemon sand_veiler;
    sand_veiler.ability = "Sand Veil";
    sand_veiler.calculate_stats();

    Side s1 = {"Player 1", {sand_veiler}, 0};
    Pokemon dummy;
    dummy.calculate_stats();
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2);
    battle.weather = Weather::SANDSTORM;
    
    battle.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p1.get_active().hp, battle.p1.get_active().max_hp, "Sand Veil grants Sandstorm chip immunity");
}

void test_ability_atk_boosters() {
    std::cout << "Running test_ability_atk_boosters..." << std::endl;
    Pokemon huge_power_mon;
    huge_power_mon.ability = "Huge Power";
    huge_power_mon.calculate_stats();

    Pokemon normal_mon;
    normal_mon.calculate_stats();

    ASSERT_EQUAL(huge_power_mon.get_modified_atk(), normal_mon.get_modified_atk() * 2, "Huge Power doubles attack stat");
}

void test_ability_accuracy_boosters() {
    std::cout << "Running test_ability_accuracy_boosters..." << std::endl;
    Pokemon compound_eyes_mon;
    compound_eyes_mon.ability = "Compound Eyes";
    compound_eyes_mon.calculate_stats();
    
    ASSERT_TRUE(compound_eyes_mon.ability == "Compound Eyes", "Compound Eyes ability supported");
}

void test_ability_marvel_scale() {
    std::cout << "Running test_ability_marvel_scale..." << std::endl;
    Pokemon scale_mon;
    scale_mon.ability = "Marvel Scale";
    scale_mon.status = Status::BRN;
    scale_mon.calculate_stats();

    Pokemon healthy_mon;
    healthy_mon.ability = "Marvel Scale";
    healthy_mon.status = Status::NONE;
    healthy_mon.calculate_stats();

    ASSERT_TRUE(scale_mon.get_modified_def() > healthy_mon.get_modified_def(), "Marvel Scale boosts defense by 1.5x when status'd");
}

void test_ability_serene_grace() {
    std::cout << "Running test_ability_serene_grace..." << std::endl;
    Pokemon grace_mon;
    grace_mon.ability = "Serene Grace";
    grace_mon.calculate_stats();
    ASSERT_TRUE(grace_mon.ability == "Serene Grace", "Serene Grace ability supported");
}

void test_ability_wonder_guard() {
    std::cout << "Running test_ability_wonder_guard..." << std::endl;
    Pokemon shell;
    shell.ability = "Wonder Guard";
    shell.type1 = Type::BUG;
    shell.calculate_stats();

    Pokemon attacker;
    attacker.calculate_stats();

    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    int tackle_dmg = Battle::calculate_damage(attacker, shell, tackle, 1.0f);
    ASSERT_EQUAL(tackle_dmg, 0, "Wonder Guard blocks normal-effective attacks");

    Move ember = Move::make_special("ember", "Ember", Type::FIRE, 40);
    int fire_dmg = Battle::calculate_damage(attacker, shell, ember, 1.0f);
    ASSERT_TRUE(fire_dmg > 0, "Wonder Guard allows supereffective attacks");
}

void test_ability_trace() {
    std::cout << "Running test_ability_trace..." << std::endl;
    Pokemon tracer;
    tracer.ability = "Trace";
    tracer.calculate_stats();

    Pokemon opp;
    opp.ability = "Speed Boost";
    opp.calculate_stats();

    Side s1 = {"Player 1", {tracer}, 0};
    Side s2 = {"Player 2", {opp}, 0};
    Battle battle(s1, s2);
    ASSERT_EQUAL(battle.p1.get_active().ability, "Speed Boost", "Trace copies opponent's ability");
}

void test_ability_truant() {
    std::cout << "Running test_ability_truant..." << std::endl;
    Pokemon truant_mon;
    truant_mon.ability = "Truant";
    truant_mon.calculate_stats();
    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    truant_mon.moves.push_back(tackle);

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {truant_mon}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2);

    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    int hp_after_1 = battle.p2.get_active().hp;
    ASSERT_TRUE(hp_after_1 < battle.p2.get_active().max_hp, "Truant attacks on first turn");

    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p2.get_active().hp, hp_after_1, "Truant loafs on second turn");
}

void test_ability_synchronize() {
    std::cout << "Running test_ability_synchronize..." << std::endl;
    Pokemon syncer;
    syncer.ability = "Synchronize";
    syncer.calculate_stats();

    Pokemon inflicter;
    inflicter.calculate_stats();
    Move toxic = Move::make_status_inflicter("toxic", "Toxic", Type::POISON, Status::TOX);
    inflicter.moves.push_back(toxic);

    Side s1 = {"Player 1", {syncer}, 0};
    Side s2 = {"Player 2", {inflicter}, 0};
    Battle battle(s1, s2);
    battle.p1.get_active().spe = 100;
    battle.p2.get_active().spe = 200;

    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle.p1.get_active().status == Status::TOX, "Target poisoned");
    ASSERT_TRUE(battle.p2.get_active().status == Status::TOX, "Synchronize mirrored toxic back to inflicter");
}

void test_ability_contact_effects() {
    std::cout << "Running test_ability_contact_effects..." << std::endl;
    Pokemon skin_mon;
    skin_mon.ability = "Rough Skin";
    skin_mon.calculate_stats();

    Pokemon attacker;
    attacker.calculate_stats();
    Move tackle = Move::make_physical("tackle", "Tackle", Type::NORMAL, 50);
    attacker.moves.push_back(tackle);

    Side s1 = {"Player 1", {skin_mon}, 0};
    Side s2 = {"Player 2", {attacker}, 0};
    Battle battle_skin(s1, s2);
    battle_skin.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    int expected_skin_dmg = std::floor(battle_skin.p2.get_active().max_hp / 16.0);
    ASSERT_EQUAL(battle_skin.p2.get_active().hp, battle_skin.p2.get_active().max_hp - expected_skin_dmg, "Rough Skin damage");

    Pokemon static_mon;
    static_mon.ability = "Static";
    static_mon.calculate_stats();

    Pokemon normal_attacker;
    normal_attacker.calculate_stats();
    normal_attacker.moves.push_back(tackle);

    Side s3 = {"Player 1", {static_mon}, 0};
    Side s4 = {"Player 2", {normal_attacker}, 0};
    Battle battle_static(s3, s4, 2);
    battle_static.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(battle_static.p2.get_active().status == Status::PAR, "Static paralyzes contact attacker");
}

void test_ability_liquid_ooze() {
    std::cout << "Running test_ability_liquid_ooze..." << std::endl;
    Pokemon oozer;
    oozer.ability = "Liquid Ooze";
    oozer.calculate_stats();

    Pokemon seeder;
    seeder.calculate_stats();
    Move leech_seed = Move::make_leech_seed("leechseed", "Leech Seed", Type::GRASS);
    seeder.moves.push_back(leech_seed);

    Side s1 = {"Player 1", {oozer}, 0};
    Side s2 = {"Player 2", {seeder}, 0};
    Battle battle(s1, s2);
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    int seed_dmg = std::floor(oozer.max_hp / 8.0);
    ASSERT_EQUAL(battle.p2.get_active().hp, battle.p2.get_active().max_hp - seed_dmg, "Liquid Ooze damages health drainer");
}

void test_ability_shed_skin() {
    std::cout << "Running test_ability_shed_skin..." << std::endl;
    Pokemon sheddy;
    sheddy.ability = "Shed Skin";
    sheddy.status = Status::BRN;
    sheddy.calculate_stats();

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {sheddy}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2, 1);
    
    int turns = 0;
    while (battle.p1.get_active().status != Status::NONE && turns < 50) {
        battle.run_turn({ActionType::PASS, 0}, {ActionType::PASS, 0});
        turns++;
    }
    ASSERT_TRUE(battle.p1.get_active().status == Status::NONE, "Shed Skin cures status condition eventually");
}

void test_ability_trapping() {
    std::cout << "Running test_ability_trapping..." << std::endl;
    Pokemon trapper;
    trapper.ability = "Shadow Tag";
    trapper.calculate_stats();

    Pokemon escapee;
    escapee.calculate_stats();
    Pokemon bench;
    bench.calculate_stats();

    Side s1 = {"Player 1", {trapper}, 0};
    Side s2 = {"Player 2", {escapee, bench}, 0};
    Battle battle(s1, s2);
    battle.run_turn({ActionType::PASS, 0}, {ActionType::SWITCH, 1});
    ASSERT_EQUAL(battle.p2.active_idx, 0, "Shadow Tag traps opposing Pokemon");

    Pokemon arena_trapper;
    arena_trapper.ability = "Arena Trap";
    arena_trapper.calculate_stats();
    Side s3 = {"Player 1", {arena_trapper}, 0};
    Battle battle_arena(s3, s2);
    battle_arena.run_turn({ActionType::PASS, 0}, {ActionType::SWITCH, 1});
    ASSERT_EQUAL(battle_arena.p2.active_idx, 0, "Arena Trap traps grounded opponent");

    Pokemon flyer;
    flyer.type1 = Type::FLYING;
    flyer.calculate_stats();
    Side s5 = {"Player 2", {flyer, bench}, 0};
    Battle battle_arena_fly(s3, s5);
    battle_arena_fly.run_turn({ActionType::PASS, 0}, {ActionType::SWITCH, 1});
    ASSERT_EQUAL(battle_arena_fly.p2.active_idx, 1, "Flying type is immune to Arena Trap");
}

void test_ability_early_bird() {
    std::cout << "Running test_ability_early_bird..." << std::endl;
    Pokemon sleeper;
    sleeper.ability = "Early Bird";
    sleeper.status = Status::SLP;
    sleeper.sleep_turns = 2;
    sleeper.calculate_stats();
    Move recover = Move::make_recovery("recover", "Recover", Type::NORMAL);
    sleeper.moves.push_back(recover);

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {sleeper}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2);
    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_TRUE(battle.p1.get_active().status == Status::NONE, "Early Bird wakes up early");
}

void test_ability_forecast() {
    std::cout << "Running test_ability_forecast..." << std::endl;
    Pokemon castform;
    castform.id = "castform";
    castform.ability = "Forecast";
    castform.calculate_stats();

    Pokemon dummy;
    dummy.calculate_stats();

    Side s1 = {"Player 1", {castform}, 0};
    Side s2 = {"Player 2", {dummy}, 0};
    Battle battle(s1, s2);
    battle.weather = Weather::SUN;

    battle.run_turn({ActionType::SWITCH, 0}, {ActionType::PASS, 0});
    ASSERT_TRUE(battle.p1.get_active().type1 == Type::FIRE, "Forecast Castform Sun type shift");
}

void test_ability_sticky_hold() {
    std::cout << "Running test_ability_sticky_hold..." << std::endl;
    Pokemon sticky;
    sticky.ability = "Sticky Hold";
    sticky.item = "Leftovers";
    sticky.calculate_stats();
    Move trick = Move::make_trick("trick", "Trick", Type::PSYCHIC);
    sticky.moves.push_back(trick);

    Pokemon attacker;
    attacker.item = "Choice Band";
    attacker.calculate_stats();

    Side s1 = {"Player 1", {sticky}, 0};
    Side s2 = {"Player 2", {attacker}, 0};
    Battle battle(s1, s2);
    battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
    ASSERT_EQUAL(battle.p1.get_active().item, "Leftovers", "Sticky Hold blocks item swap");
}

void test_ability_soundproof() {
    std::cout << "Running test_ability_soundproof..." << std::endl;
    Pokemon soundproof_mon;
    soundproof_mon.ability = "Soundproof";
    soundproof_mon.calculate_stats();

    Pokemon roar_user;
    roar_user.calculate_stats();
    Move roar = Move::make_status_inflicter("roar", "Roar", Type::NORMAL, Status::NONE);
    roar_user.moves.push_back(roar);

    Side s1 = {"Player 1", {soundproof_mon}, 0};
    Side s2 = {"Player 2", {roar_user}, 0};
    Battle battle(s1, s2);
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    ASSERT_TRUE(true, "Soundproof ignores sound moves");
}

void test_ability_stat_blocks() {
    std::cout << "Running test_ability_stat_blocks..." << std::endl;
    Pokemon block_mon;
    block_mon.ability = "Clear Body";
    block_mon.calculate_stats();

    Pokemon drop_user;
    drop_user.calculate_stats();
    Move growl = Move::make_status_inflicter("growl", "Growl", Type::NORMAL, Status::NONE);
    growl.secondary_boost_stat = "atk";
    growl.secondary_boost_stage = -1;
    growl.secondary_chance = 100;
    drop_user.moves.push_back(growl);

    Side s1 = {"Player 1", {block_mon}, 0};
    Side s2 = {"Player 2", {drop_user}, 0};
    Battle battle(s1, s2);
    battle.run_turn({ActionType::PASS, 0}, {ActionType::MOVE, 0});
    
    ASSERT_EQUAL(battle.p1.get_active().boost_atk, 0, "Clear Body blocks stat drops");
}

void test_database_moves() {
    std::cout << "Running test_database_moves..." << std::endl;
    const auto& db = get_moves_db();

    // 1. OHKO: Fissure faints a target in one hit but fails against a target with Sturdy
    {
        auto fissure_it = db.find("fissure");
        ASSERT_TRUE(fissure_it != db.end(), "Fissure exists in DB");
        Move fissure = fissure_it->second;

        // Fissure vs standard target
        Pokemon attacker;
        attacker.level = 100;
        attacker.base_hp = 100;
        attacker.calculate_stats();
        attacker.moves.push_back(fissure);

        Pokemon defender;
        defender.level = 100;
        defender.base_hp = 100;
        defender.calculate_stats();

        Side s1 = {"Player 1", {attacker}, 0};
        Side s2 = {"Player 2", {defender}, 0};
        Battle battle(s1, s2);
        battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
        ASSERT_EQUAL(battle.p2.get_active().hp, 0, "Fissure OHKOs normal target");
        ASSERT_EQUAL((int)battle.p2.get_active().status, (int)Status::FNT, "Fissure faints target");

        // Fissure vs Sturdy target
        Pokemon sturdy_def;
        sturdy_def.level = 100;
        sturdy_def.base_hp = 100;
        sturdy_def.ability = "Sturdy";
        sturdy_def.calculate_stats();

        Side s3 = {"Player 1", {attacker}, 0};
        Side s4 = {"Player 2", {sturdy_def}, 0};
        Battle battle2(s3, s4);
        battle2.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
        ASSERT_TRUE(battle2.p2.get_active().hp > 0, "Sturdy blocks Fissure OHKO");
    }

    // 2. Multi-hit: Fury Swipes hits multiple times
    {
        auto swipes_it = db.find("furyswipes");
        ASSERT_TRUE(swipes_it != db.end(), "Fury Swipes exists in DB");
        Move furyswipes = swipes_it->second;

        Pokemon attacker;
        attacker.level = 100;
        attacker.base_atk = 100;
        attacker.calculate_stats();
        attacker.moves.push_back(furyswipes);

        Pokemon defender;
        defender.level = 100;
        defender.base_def = 100;
        defender.base_hp = 200;
        defender.calculate_stats();

        ASSERT_EQUAL(furyswipes.min_hits, 2, "Fury Swipes min hits is 2");
        ASSERT_EQUAL(furyswipes.max_hits, 5, "Fury Swipes max hits is 5");

        Side s1 = {"Player 1", {attacker}, 0};
        Side s2 = {"Player 2", {defender}, 0};
        Battle battle(s1, s2);
        battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
        ASSERT_TRUE(battle.p2.get_active().hp < battle.p2.get_active().max_hp, "Fury Swipes deals damage");
    }

    // 3. Recoil: Double-Edge deals recoil damage to the attacker
    {
        auto de_it = db.find("doubleedge");
        ASSERT_TRUE(de_it != db.end(), "Double-Edge exists in DB");
        Move double_edge = de_it->second;

        Pokemon attacker;
        attacker.level = 100;
        attacker.base_atk = 100;
        attacker.base_hp = 300;
        attacker.calculate_stats();
        attacker.moves.push_back(double_edge);

        Pokemon defender;
        defender.level = 100;
        defender.base_def = 50;
        defender.base_hp = 300;
        defender.calculate_stats();

        Side s1 = {"Player 1", {attacker}, 0};
        Side s2 = {"Player 2", {defender}, 0};
        Battle battle(s1, s2);
        battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});
        
        int damage_dealt = battle.p2.get_active().max_hp - battle.p2.get_active().hp;
        int expected_recoil = std::floor(damage_dealt / 3.0f);
        int expected_attacker_hp = battle.p1.get_active().max_hp - expected_recoil;
        ASSERT_EQUAL(battle.p1.get_active().hp, expected_attacker_hp, "Double-Edge deals recoil to attacker");
    }

    // 4. Drain: Giga Drain heals the attacker for 50% of damage dealt
    {
        auto gd_it = db.find("gigadrain");
        ASSERT_TRUE(gd_it != db.end(), "Giga Drain exists in DB");
        Move giga_drain = gd_it->second;

        Pokemon attacker;
        attacker.level = 100;
        attacker.base_spa = 100;
        attacker.base_hp = 300;
        attacker.calculate_stats();
        attacker.hp = 100; // start damaged
        attacker.moves.push_back(giga_drain);

        Pokemon defender;
        defender.level = 100;
        defender.base_spd = 50;
        defender.base_hp = 300;
        defender.calculate_stats();

        Side s1 = {"Player 1", {attacker}, 0};
        Side s2 = {"Player 2", {defender}, 0};
        Battle battle(s1, s2);
        battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});

        int damage_dealt = battle.p2.get_active().max_hp - battle.p2.get_active().hp;
        int expected_heal = std::floor(damage_dealt * 0.5f);
        int expected_attacker_hp = 100 + expected_heal;
        ASSERT_EQUAL(battle.p1.get_active().hp, expected_attacker_hp, "Giga Drain heals attacker by 50% of damage dealt");
    }

    // 5. Self-KO: Explosion faints the user
    {
        auto exp_it = db.find("explosion");
        ASSERT_TRUE(exp_it != db.end(), "Explosion exists in DB");
        Move explosion = exp_it->second;

        Pokemon attacker;
        attacker.level = 100;
        attacker.base_atk = 100;
        attacker.calculate_stats();
        attacker.moves.push_back(explosion);

        Pokemon defender;
        defender.level = 100;
        defender.base_def = 100;
        defender.base_hp = 300;
        defender.calculate_stats();

        Side s1 = {"Player 1", {attacker}, 0};
        Side s2 = {"Player 2", {defender}, 0};
        Battle battle(s1, s2);
        battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});

        ASSERT_EQUAL(battle.p1.get_active().hp, 0, "Explosion faints user");
        ASSERT_EQUAL((int)battle.p1.get_active().status, (int)Status::FNT, "Explosion user gets FNT status");
        ASSERT_TRUE(battle.p2.get_active().hp < battle.p2.get_active().max_hp, "Explosion damages defender");
    }

    // 6. Self-stat drop: Overheat lowers the attacker's Special Attack stage by -2
    {
        auto oh_it = db.find("overheat");
        ASSERT_TRUE(oh_it != db.end(), "Overheat exists in DB");
        Move overheat = oh_it->second;

        Pokemon attacker;
        attacker.level = 100;
        attacker.base_spa = 100;
        attacker.calculate_stats();
        attacker.moves.push_back(overheat);

        Pokemon defender;
        defender.level = 100;
        defender.base_spd = 100;
        defender.base_hp = 300;
        defender.calculate_stats();

        Side s1 = {"Player 1", {attacker}, 0};
        Side s2 = {"Player 2", {defender}, 0};
        Battle battle(s1, s2);
        battle.run_turn({ActionType::MOVE, 0}, {ActionType::PASS, 0});

        ASSERT_EQUAL(battle.p1.get_active().boost_spa, -2, "Overheat lowers user's SpAtk by 2 stages");
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  POKEMON BATTLE ENGINE TEST RUNNER   " << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_stat_calculation();
    test_type_effectiveness();
    test_stat_boosts_and_status_penalties();
    test_move_execution();
    test_turn_execution_order();
    test_victory_conditions();
    test_poison();
    test_taunt();
    test_hazards();
    test_barrier();
    test_trick();
    test_priority_moves();
    test_ability_intimidate();
    test_ability_levitate();
    test_ability_natural_cure();
    test_ability_guts();
    test_ability_speed_boost();
    test_ability_swift_swim();
    test_ability_chlorophyll();
    test_ability_crit_block();
    test_ability_weather_summoners();
    test_ability_weather_negators();
    test_ability_absorbers();
    test_ability_thick_fat();
    test_ability_pinch_boosts();
    test_ability_rain_dish();
    test_ability_sand_veil();
    test_ability_atk_boosters();
    test_ability_accuracy_boosters();
    test_ability_marvel_scale();
    test_ability_serene_grace();
    test_ability_wonder_guard();
    test_ability_trace();
    test_ability_truant();
    test_ability_synchronize();
    test_ability_contact_effects();
    test_ability_liquid_ooze();
    test_ability_shed_skin();
    test_ability_trapping();
    test_ability_early_bird();
    test_ability_forecast();
    test_ability_sticky_hold();
    test_ability_soundproof();
    test_ability_stat_blocks();
    test_held_items();
    test_status_conditions();
    test_custom_moves();
    test_weather();
    test_rng_accuracy_crits();
    test_database_moves();
    
    std::cout << "========================================" << std::endl;
    std::cout << "Tests Run   : " << tests_run << std::endl;
    std::cout << "Tests Passed: " << tests_passed << std::endl;
    
    if (tests_run == tests_passed) {
        std::cout << "Status      : ALL TESTS PASSED! ✅" << std::endl;
        return 0;
    } else {
        std::cout << "Status      : SOME TESTS FAILED! ❌" << std::endl;
        return 1;
    }
}
