let currentState = null;
let autoRefresh = null;
let currentValidActions = [];

function getSpriteUrl(name, isBack=false) {
    // Showdown sprite URLs
    // For simplicity, we just use gen5ani which has great animated pixel sprites
    if (!name) return '';
    let cleanName = name.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (isBack) {
        return `https://play.pokemonshowdown.com/sprites/gen5ani-back/${cleanName}.gif`;
    }
    return `https://play.pokemonshowdown.com/sprites/gen5ani/${cleanName}.gif`;
}

function updateHpBar(elementId, hp, maxHp) {
    const fill = document.getElementById(elementId);
    if (!fill) return;
    
    let percent = 0;
    if (maxHp > 0) percent = (hp / maxHp) * 100;
    
    fill.style.width = `${percent}%`;
    
    // Colors
    fill.classList.remove('yellow', 'red');
    if (percent <= 20) {
        fill.classList.add('red');
    } else if (percent <= 50) {
        fill.classList.add('yellow');
    }
}

function updateStatus(elementId, status) {
    const el = document.getElementById(elementId);
    el.className = 'status-badge hidden'; // reset
    if (status && status !== '') {
        el.textContent = status.toUpperCase();
        el.classList.add(`status-${status}`);
        el.classList.remove('hidden');
    }
}

function renderPartyBalls(count) {
    const container = document.getElementById('p2PartyBalls');
    container.innerHTML = '';
    for (let i = 0; i < 6; i++) {
        const div = document.createElement('div');
        div.className = 'ball-icon';
        if (i >= count) {
            div.classList.add('fainted');
        }
        container.appendChild(div);
    }
}

function renderBench(benchData, validActions) {
    const grid = document.getElementById('benchGrid');
    grid.innerHTML = '';
    
    benchData.forEach((p, idx) => {
        const isFainted = p.hp === 0;
        let percent = (p.hp / p.max_hp) * 100;
        let colorClass = '';
        if (percent <= 20) colorClass = 'background: #f84838;';
        else if (percent <= 50) colorClass = 'background: #f8e838;';
        else colorClass = 'background: #48f858;';

        const actionMatch = validActions.find(a => a.id.startsWith(`switch ${idx+2}`));
        const canSwitch = actionMatch && !isFainted;

        const card = document.createElement('div');
        card.className = `bench-card ${isFainted ? 'fainted' : ''}`;
        const itemText = p.item ? p.item : 'None';
        const movesText = p.moves ? p.moves.join(', ') : 'Unknown';
        
        card.innerHTML = `
            <div class="bench-name">${p.display_name}</div>
            <div class="bench-hp-bar">
                <div style="height: 100%; width: ${percent}%; ${colorClass}"></div>
            </div>
            <div style="font-size: 10px; margin-bottom: 4px;">HP: ${p.hp}/${p.max_hp}
                ${p.status ? `<span class="status-badge status-${p.status}">${p.status.toUpperCase()}</span>` : ''}
            </div>
            <div style="font-size: 9px; color: #555;">Item: ${itemText}</div>
            <div style="font-size: 8px; color: #666; margin-top: 4px;">Moves: ${movesText}</div>
        `;
        
        if (canSwitch) {
            card.onclick = () => {
                closeBench();
                sendAction(actionMatch.id);
            };
        } else {
            card.style.opacity = '0.5';
            card.style.cursor = 'not-allowed';
        }
        
        grid.appendChild(card);
    });
}

const typeColors = {
    "Normal": "#A8A878", "Fire": "#F08030", "Water": "#6890F0",
    "Electric": "#F8D030", "Grass": "#78C850", "Ice": "#98D8D8",
    "Fighting": "#C03028", "Poison": "#A040A0", "Ground": "#E0C068",
    "Flying": "#A890F0", "Psychic": "#F85888", "Bug": "#A8B820",
    "Rock": "#B8A038", "Ghost": "#705898", "Dragon": "#7038F8",
    "Dark": "#705848", "Steel": "#B8B8D0", "Fairy": "#EE99AC"
};

function renderMovesMenu(moves, validActions) {
    const menu = document.getElementById('movesMenu');
    menu.innerHTML = '';
    
    moves.forEach((m, idx) => {
        const btn = document.createElement('button');
        btn.className = 'move-btn';
        
        let powerStr = m.category !== 'Status' && m.basePower > 0 ? ` PWR:${m.basePower}` : '';
        const color = typeColors[m.type] || "#f8f8f8";
        
        // Find corresponding action
        const actionMatch = validActions.find(a => a.id.startsWith(`move ${idx+1}`));
        
        if (actionMatch) {
            btn.innerHTML = `${m.move}<br><span style="font-size: 8px;">PP ${m.pp}/${m.maxpp}${powerStr}</span>`;
            btn.style.backgroundColor = color;
            // ensure text is readable
            btn.style.color = ["Electric", "Ice", "Normal", "Flying", "Bug", "Steel"].includes(m.type) ? "#222" : "#fff";
            btn.onclick = () => {
                backToMenu();
                sendAction(actionMatch.id);
            };
        } else {
            btn.innerHTML = `${m.move}<br><span style="font-size: 8px;">-</span>`;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
            btn.style.backgroundColor = color;
        }
        menu.appendChild(btn);
    });
    
    // Add cancel button if there's an odd number of moves or to complete the grid
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'move-btn';
    cancelBtn.textContent = 'CANCEL';
    cancelBtn.style.gridColumn = 'span 2'; // Prevents layout from shifting
    cancelBtn.style.backgroundColor = '#ccc';
    cancelBtn.style.color = '#000';
    cancelBtn.onclick = backToMenu;
    menu.appendChild(cancelBtn);
}

let currentLogLength = 0;

function parseLogLine(line) {
    const parts = line.split('|');
    if (parts.length < 2) return null;
    
    const type = parts[1];
    
    // Helper to get pure pokemon name (e.g., "p1a: Pikachu" -> "Pikachu")
    const getName = (str) => {
        if (!str) return "";
        const s = str.split(': ');
        return s.length > 1 ? s[1] : s[0];
    };
    
    switch(type) {
        case 'move':
            return `${getName(parts[2])} used ${parts[3]}!`;
        case '-supereffective':
            return `It's super effective!`;
        case '-resisted':
            return `It's not very effective...`;
        case '-crit':
            return `A critical hit!`;
        case 'faint':
            return `${getName(parts[2])} fainted!`;
        case 'switch':
        case 'drag':
            return `${parts[2].startsWith('p1') ? 'You' : 'Opponent'} sent out ${getName(parts[2])}!`;
        case '-status':
            return `${getName(parts[2])} was afflicted with ${parts[3].toUpperCase()}!`;
        case '-curestatus':
            return `${getName(parts[2])} was cured of its status!`;
        case '-boost':
            return `${getName(parts[2])}'s ${parts[3]} rose!`;
        case '-unboost':
            return `${getName(parts[2])}'s ${parts[3]} fell!`;
        case '-damage':
            if (parts.length > 4 && parts[4].includes('[from]')) {
                if (parts[4].includes('psn')) return `${getName(parts[2])} was hurt by poison!`;
                if (parts[4].includes('brn')) return `${getName(parts[2])} was hurt by its burn!`;
                if (parts[4].includes('confusion')) return `${getName(parts[2])} hurt itself in its confusion!`;
                if (parts[4].includes('Spikes')) return `${getName(parts[2])} was hurt by spikes!`;
                if (parts[4].includes('Stealth Rock')) return `Pointed stones dug into ${getName(parts[2])}!`;
                if (parts[4].includes('Sandstorm')) return `${getName(parts[2])} is buffeted by the sandstorm!`;
                if (parts[4].includes('Hail')) return `${getName(parts[2])} is buffeted by the hail!`;
            }
            return null; // Ignore normal damage, HP bar shows it
        case '-heal':
            if (parts.length > 4 && parts[4].includes('Leftovers')) {
                return `${getName(parts[2])} restored HP using its Leftovers!`;
            }
            return null; // Ignore general heal, HP bar shows it
        case '-start':
            if (parts[3].includes('confusion')) return `${getName(parts[2])} became confused!`;
            if (parts[3].includes('Leech Seed')) return `${getName(parts[2])} was seeded!`;
            return null;
        case '-end':
            if (parts[3].includes('confusion')) return `${getName(parts[2])} snapped out of its confusion!`;
            return null;
        case 'cant':
            if (parts.length > 3) {
                if (parts[3] === 'par') return `${getName(parts[2])} is paralyzed! It can't move!`;
                if (parts[3] === 'flinch') return `${getName(parts[2])} flinched and couldn't move!`;
                if (parts[3] === 'slp') return `${getName(parts[2])} is fast asleep.`;
                if (parts[3] === 'frz') return `${getName(parts[2])} is frozen solid!`;
            }
            return `${getName(parts[2])} can't move!`;
        case '-fail':
            return `But it failed!`;
        case '-miss':
            return `The attack missed!`;
        case 'win':
            return `Player ${parts[2] === 'p1' ? '1 (You)' : '2 (AI)'} wins!`;
        case 'tie':
            return `It's a tie!`;
        case '-weather':
            if (parts[2] === 'none' || parts[2] === 'Upkeep') return null;
            return `The weather changed to ${parts[2]}!`;
        default:
            return null; // Ignore raw stats and complex states
    }
}

function typeLog(lines) {
    if (lines.length === currentLogLength) return;
    
    const box = document.getElementById('dialogBox');
    box.innerHTML = '';
    
    // Parse into friendly text, keeping only valid lines
    const friendlyLines = [];
    lines.forEach(l => {
        const parsed = parseLogLine(l);
        if (parsed) friendlyLines.push(parsed);
    });
    
    friendlyLines.forEach(l => {
        const div = document.createElement('div');
        div.textContent = l;
        div.style.marginBottom = "4px";
        box.appendChild(div);
    });
    box.scrollTop = box.scrollHeight;
    
    currentLogLength = lines.length;
}

function showMoves() {
    document.getElementById('mainMenu').style.display = 'none';
    document.getElementById('movesMenu').style.display = 'grid';
}

function showBench() {
    document.getElementById('benchOverlay').style.display = 'block';
}

function closeBench() {
    document.getElementById('benchOverlay').style.display = 'none';
}

function backToMenu() {
    document.getElementById('movesMenu').style.display = 'none';
    document.getElementById('mainMenu').style.display = 'grid';
}

function triggerShake(elementId) {
    const el = document.getElementById(elementId);
    el.classList.remove('shake');
    void el.offsetWidth; // trigger reflow
    el.classList.add('shake');
}

async function updateUI(data) {
    // P1
    if (data.p1_active) {
        document.getElementById('p1Name').textContent = data.p1_active.display_name;
        document.getElementById('p1HpText').textContent = `${data.p1_active.hp}/${data.p1_active.max_hp}`;
        document.getElementById('p1Item').textContent = data.p1_active.item ? data.p1_active.item : 'None';
        updateHpBar('p1HpFill', data.p1_active.hp, data.p1_active.max_hp);
        updateStatus('p1Status', data.p1_active.status);
        
        const p1Sprite = document.getElementById('spritePlayer');
        const newSprite = getSpriteUrl(data.p1_active.name, true);
        if (!p1Sprite.style.backgroundImage.includes(newSprite)) {
            p1Sprite.style.backgroundImage = `url(${newSprite})`;
        }
    }
    
    // P2
    if (data.p2_active) {
        document.getElementById('p2Name').textContent = data.p2_active.display_name;
        updateHpBar('p2HpFill', data.p2_active.hp, data.p2_active.max_hp);
        updateStatus('p2Status', data.p2_active.status);
        
        const p2Sprite = document.getElementById('spriteOpponent');
        const newSprite2 = getSpriteUrl(data.p2_active.name, false);
        if (!p2Sprite.style.backgroundImage.includes(newSprite2)) {
            p2Sprite.style.backgroundImage = `url(${newSprite2})`;
        }
    }
    
    renderPartyBalls(data.p2_party_count);
    
    if (data.p1_bench && document.getElementById('benchOverlay').style.display !== 'block') {
        renderBench(data.p1_bench, data.valid_actions);
    }
    
    if (data.p1_moves && document.getElementById('movesMenu').style.display !== 'grid') {
        renderMovesMenu(data.p1_moves, data.valid_actions);
    }
    
    if (data.log && data.log.length > 0) {
        typeLog(data.log);
        const lastLog = data.log[data.log.length - 1];
        if (lastLog.includes('|-damage|p1a')) triggerShake('spritePlayer');
        if (lastLog.includes('|-damage|p2a')) triggerShake('spriteOpponent');
    } else if (!data.is_thinking && !data.is_terminal) {
        const activeName = data.p1_active ? data.p1_active.display_name.toUpperCase() : 'POKéMON';
        document.getElementById('dialogBox').innerHTML = `What will ${activeName} do?`;
    }

    if (data.is_terminal) {
        document.getElementById('dialogBox').innerHTML = 'Battle ended.';
        document.getElementById('mainMenu').innerHTML = `
            <div style="grid-column: span 2; text-align: center; color: ${data.winner === 'p1' ? 'green' : 'red'};">
                Game Over! Winner: ${data.winner === 'p1' ? 'YOU' : 'AI'}
            </div>
            <button class="menu-btn" onclick="resetGame()" style="grid-column: span 2;">PLAY AGAIN</button>
        `;
        document.getElementById('mainMenu').style.display = 'grid';
        document.getElementById('movesMenu').style.display = 'none';
    } else if (data.is_thinking) {
        document.getElementById('mainMenu').style.display = 'none';
        document.getElementById('movesMenu').style.display = 'none';
        document.getElementById('dialogBox').innerHTML = 'Waiting for AI<span class="dots"></span>';
    } else {
        if (data.valid_actions && data.valid_actions.length === 1 && data.valid_actions[0].id === "pass") {
            sendAction("pass");
            return;
        }
        
        if (document.getElementById('movesMenu').style.display !== 'grid' && 
            document.getElementById('benchOverlay').style.display !== 'block') {
            document.getElementById('mainMenu').style.display = 'grid';
        }
    }
}

async function startPolling() {
    while (true) {
        try {
            const res = await fetch('/state?t=' + new Date().getTime(), { cache: 'no-store' });
            const data = await res.json();
            if (!data.error) {
                currentState = data;
                updateUI(data);
                if (data.is_terminal) break;
            }
        } catch(e) {
            console.error("Polling error:", e);
        }
        await new Promise(r => setTimeout(r, 1000));
    }
}

async function sendAction(actionStr) {
    document.getElementById('mainMenu').style.display = 'none';
    document.getElementById('movesMenu').style.display = 'none';
    document.getElementById('dialogBox').innerHTML = 'Waiting for AI<span class="dots"></span>';
    
    try {
        await fetch('/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionStr })
        });
    } catch(e) {
        console.error("Action error:", e);
    }
}

async function resetGame() {
    await fetch('/reset', { method: 'POST' });
    location.reload();
}

// Init
startPolling();
