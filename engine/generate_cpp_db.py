import json

with open("gen3_moves.json", "r") as f:
    moves = json.load(f)

# Let's inspect the types of effects in moves
print(f"Total moves loaded: {len(moves)}")

# Types list mapping to C++ enums
type_map = {
    "Normal": "Type::NORMAL", "Fire": "Type::FIRE", "Water": "Type::WATER",
    "Grass": "Type::GRASS", "Electric": "Type::ELECTRIC", "Ice": "Type::ICE",
    "Fighting": "Type::FIGHTING", "Poison": "Type::POISON", "Ground": "Type::GROUND",
    "Flying": "Type::FLYING", "Psychic": "Type::PSYCHIC", "Bug": "Type::BUG",
    "Rock": "Type::ROCK", "Ghost": "Type::GHOST", "Dragon": "Type::DRAGON",
    "Steel": "Type::STEEL", "Dark": "Type::DARK", "???": "Type::NONE"
}

cat_map = {
    "Physical": "Category::PHYSICAL",
    "Special": "Category::SPECIAL",
    "Status": "Category::STATUS"
}

status_map = {
    "tox": "Status::TOX", "brn": "Status::BRN", "par": "Status::PAR",
    "slp": "Status::SLP", "frz": "Status::FRZ", "fnt": "Status::FNT"
}

# We want to identify and map the move effects:
# Let's write the C++ code generator for move_db.h
out = []
out.append("#ifndef MOVE_DB_H")
out.append("#define MOVE_DB_H")
out.append("")
out.append('#include "move.h"')
out.append('#include <unordered_map>')
out.append('#include <string>')
out.append("")
out.append("inline const std::unordered_map<std::string, Move>& get_moves_db() {")
out.append("    static const std::unordered_map<std::string, Move> db = {")

for m in moves:
    mid = m['id']
    name = m['name']
    mtype = type_map.get(m['type'], "Type::NONE")
    mcat = cat_map.get(m['category'], "Category::STATUS")
    power = m['basePower']
    acc = m['accuracy']
    if isinstance(acc, str) or acc is True or acc == 0:
        acc = 0 # always hits
    pri = m['priority']

    # Determine move effect
    effect = "MoveEffect::NONE"
    
    # We can detect and map custom effects from our existing enums:
    if mid == "recover" or mid == "softboiled" or mid == "milkdrink" or mid == "slackoff" or mid == "roost":
        effect = "MoveEffect::RECOVERY"
    elif mid == "taunt":
        effect = "MoveEffect::TAUNT"
    elif mid == "spikes":
        effect = "MoveEffect::HAZARDS"
    elif mid == "trick":
        effect = "MoveEffect::TRICK"
    elif mid == "protect" or mid == "detect":
        effect = "MoveEffect::PROTECT"
    elif mid == "substitute":
        effect = "MoveEffect::SUBSTITUTE"
    elif mid == "leechseed":
        effect = "MoveEffect::LEECH_SEED"
    elif mid == "destinybond":
        effect = "MoveEffect::DESTINY_BOND"
    elif mid == "rapidspin":
        effect = "MoveEffect::RAPID_SPIN"
    elif mid == "sunnyday":
        effect = "MoveEffect::WEATHER_SUN"
    elif mid == "raindance":
        effect = "MoveEffect::WEATHER_RAIN"
    elif mid == "sandstorm":
        effect = "MoveEffect::WEATHER_SAND"
    elif mid == "hail":
        effect = "MoveEffect::WEATHER_HAIL"
    elif m['category'] == "Status" and m.get('boosts'):
        effect = "MoveEffect::STAT_BOOST"
    elif m['category'] == "Status" and m.get('status'):
        effect = "MoveEffect::STATUS_MOVE"

    # Stat boosts
    batk = bdef = bspa = bspd = bspe = 0
    if m.get('boosts'):
        b = m['boosts']
        batk = b.get('atk', 0)
        bdef = b.get('def', 0)
        bspa = b.get('spa', 0)
        bspd = b.get('spd', 0)
        bspe = b.get('spe', 0)

    # Status to inflict
    status_inflict = "Status::NONE"
    if m.get('status'):
        status_inflict = status_map.get(m['status'], "Status::NONE")

    # Secondary effects
    sec_chance = 0
    sec_status = "Status::NONE"
    sec_boost_stage = 0
    sec_boost_stat = ""

    if m.get('secondary') and m['secondary']:
        sec = m['secondary']
        sec_chance = sec.get('chance', 0)
        if sec_chance is None:
            sec_chance = 100
        if sec.get('status'):
            sec_status = status_map.get(sec['status'], "Status::NONE")
        if sec.get('boosts'):
            sb = sec['boosts']
            for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']:
                if stat in sb:
                    sec_boost_stat = "acc" if stat == "accuracy" else ("eva" if stat == "evasion" else stat)
                    sec_boost_stage = sb[stat]
                    break

    # Self-stat drop/boost (e.g. Overheat)
    self_boost_stage = 0
    self_boost_stat = ""
    if m.get('self') and m['self'] and m['self'].get('boosts'):
        sb = m['self']['boosts']
        for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']:
            if stat in sb:
                self_boost_stat = "acc" if stat == "accuracy" else ("eva" if stat == "evasion" else stat)
                self_boost_stage = sb[stat]
                break

    # Drain and Recoil
    drain_factor = 0.0
    if m.get('drain'):
        d = m['drain']
        drain_factor = d[0] / d[1]

    recoil_factor = 0.0
    if m.get('recoil'):
        r = m['recoil']
        recoil_factor = r[0] / r[1]

    # OHKO and Self-KO
    self_ko = "false"
    if mid == "selfdestruct" or mid == "explosion" or mid == "memento":
        self_ko = "true"

    ohko = "false"
    if mid == "fissure" or mid == "guillotine" or mid == "horndrill" or mid == "sheercold":
        ohko = "true"

    # Multi-hit
    min_hits = 1
    max_hits = 1
    if m.get('multihit'):
        mh = m['multihit']
        if isinstance(mh, list):
            min_hits = mh[0]
            max_hits = mh[1]
        else:
            min_hits = max_hits = mh

    # Escape quotes in names
    escaped_name = name.replace('"', '\\"')

    cpp_line = (
        f'        {{"{mid}", Move{{"{mid}", "{escaped_name}", {mtype}, {mcat}, {power}, {acc}, {pri}, {effect}, '
        f'{batk}, {bdef}, {bspa}, {bspd}, {bspe}, {status_inflict}, '
        f'{sec_chance}, {sec_status}, {sec_boost_stage}, "{sec_boost_stat}", '
        f'{self_boost_stage}, "{self_boost_stat}", {drain_factor}f, {recoil_factor}f, {self_ko}, {ohko}, {min_hits}, {max_hits}}}}},'
    )
    out.append(cpp_line)

out.append("    };")
out.append("    return db;")
out.append("}")
out.append("")
out.append("#endif // MOVE_DB_H")

with open("../new_engine/move_db.h", "w") as f:
    f.write("\n".join(out))

print("C++ move database file generated: new_engine/move_db.h")
