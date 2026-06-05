#ifndef TYPES_H
#define TYPES_H

#include <string>

// Gen 3 Type List (Fairy does not exist in Gen 3)
enum class Type {
    NORMAL, FIRE, WATER, GRASS, ELECTRIC, ICE, FIGHTING, POISON, GROUND,
    FLYING, PSYCHIC, BUG, ROCK, GHOST, DRAGON, STEEL, DARK, NONE
};

inline std::string type_to_string(Type t) {
    switch(t) {
        case Type::NORMAL:   return "Normal";
        case Type::FIRE:     return "Fire";
        case Type::WATER:    return "Water";
        case Type::GRASS:    return "Grass";
        case Type::ELECTRIC: return "Electric";
        case Type::ICE:      return "Ice";
        case Type::FIGHTING: return "Fighting";
        case Type::POISON:   return "Poison";
        case Type::GROUND:   return "Ground";
        case Type::FLYING:   return "Flying";
        case Type::PSYCHIC:  return "Psychic";
        case Type::BUG:      return "Bug";
        case Type::ROCK:     return "Rock";
        case Type::GHOST:    return "Ghost";
        case Type::DRAGON:   return "Dragon";
        case Type::STEEL:    return "Steel";
        case Type::DARK:     return "Dark";
        default:             return "None";
    }
}

// Move category (Status moves do no direct damage)
enum class Category {
    PHYSICAL, SPECIAL, STATUS
};

enum class Weather {
    NONE, SUN, RAIN, SANDSTORM, HAIL
};

// Status conditions
enum class Status {
    NONE, TOX, BRN, PAR, SLP, FRZ, FNT
};

inline std::string status_to_string(Status s) {
    switch(s) {
        case Status::TOX:  return "TOX";
        case Status::BRN:  return "BRN";
        case Status::PAR:  return "PAR";
        case Status::SLP:  return "SLP";
        case Status::FRZ:  return "FRZ";
        case Status::FNT:  return "FNT";
        default:           return "NONE";
    }
}

// Turn action choices
enum class ActionType {
    MOVE, SWITCH, PASS
};

struct Action {
    ActionType type = ActionType::PASS;
    int index = 0; // Move slot (0-3) or switch target index (0-5)
};

#endif // TYPES_H
