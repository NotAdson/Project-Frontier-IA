const SPRITE_BASE_URL = "https://play.pokemonshowdown.com/sprites/";

function getSpriteUrl(pokemonName, isBack) {
    const dir = isBack ? "ani-back" : "ani";
    return `${SPRITE_BASE_URL}${dir}/${pokemonName}.gif`;
}

function updateHpBar(elementId, hp, maxHp) {
    const percentage = Math.max(0, Math.min(100, (hp / maxHp) * 100));
    const bar = document.getElementById(elementId);
    bar.style.width = `${percentage}%`;
    
    if (percentage > 50) {
        bar.style.backgroundColor = "#2ecc71"; // Green
    } else if (percentage > 20) {
        bar.style.backgroundColor = "#f1c40f"; // Yellow
    } else {
        bar.style.backgroundColor = "#e74c3c"; // Red
    }
}

async function fetchState() {
    try {
        const response = await fetch('/state');
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
            return;
        }
        
        renderState(data);
    } catch (err) {
        console.error("Failed to fetch state:", err);
    }
}

function renderState(data) {
    // Render P2 (Opponent)
    if (data.p2_active) {
        document.getElementById('p2-name').innerText = data.p2_active.display_name + (data.p2_active.status ? ` [${data.p2_active.status}]` : "");
        document.getElementById('p2-hp-text').innerText = `${data.p2_active.hp}/${data.p2_active.max_hp}`;
        updateHpBar('p2-hp-bar', data.p2_active.hp, data.p2_active.max_hp);
        
        const p2Sprite = document.getElementById('p2-sprite');
        p2Sprite.src = getSpriteUrl(data.p2_active.name, false);
        p2Sprite.style.display = "block";
    }

    // Render P1 (Player)
    if (data.p1_active) {
        document.getElementById('p1-name').innerText = data.p1_active.display_name + (data.p1_active.status ? ` [${data.p1_active.status}]` : "");
        document.getElementById('p1-hp-text').innerText = `${data.p1_active.hp}/${data.p1_active.max_hp}`;
        updateHpBar('p1-hp-bar', data.p1_active.hp, data.p1_active.max_hp);
        
        const p1Sprite = document.getElementById('p1-sprite');
        p1Sprite.src = getSpriteUrl(data.p1_active.name, true);
        p1Sprite.style.display = "block";
    }

    // Render Log
    const logPanel = document.getElementById('log-panel');
    logPanel.innerHTML = "";
    if (data.log && data.log.length > 0) {
        data.log.forEach(line => {
            const p = document.createElement('div');
            // Basic formatting to make showdown logs slightly cleaner
            let cleanLine = line.replace(/\|/g, "  ").trim();
            p.innerText = cleanLine;
            logPanel.appendChild(p);
        });
        logPanel.scrollTop = logPanel.scrollHeight;
    }

    // Render Actions or Game Over
    const actionsContainer = document.getElementById('actions-container');
    const loading = document.getElementById('loading');
    const gameOver = document.getElementById('game-over');
    
    actionsContainer.innerHTML = "";
    loading.classList.add('hidden');
    
    if (data.is_terminal) {
        actionsContainer.classList.add('hidden');
        gameOver.classList.remove('hidden');
        document.getElementById('winner-text').innerText = `Winner: ${data.winner}`;
    } else {
        actionsContainer.classList.remove('hidden');
        gameOver.classList.add('hidden');
        
        if (data.valid_actions) {
            data.valid_actions.forEach(actionObj => {
                const btn = document.createElement('button');
                btn.className = 'action-btn';
                if (actionObj.id.startsWith('switch')) {
                    btn.classList.add('switch-btn');
                }
                btn.innerText = actionObj.text;
                btn.onclick = () => submitAction(actionObj.id);
                actionsContainer.appendChild(btn);
            });
        }
    }
}

async function submitAction(action) {
    // Hide actions and show loading
    document.getElementById('actions-container').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');
    
    try {
        await fetch('/action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ action: action })
        });
        
        // Fetch new state after action is processed
        fetchState();
    } catch (err) {
        console.error("Failed to submit action:", err);
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('actions-container').classList.remove('hidden');
    }
}

async function resetGame() {
    await fetch('/reset', { method: 'POST' });
    fetchState();
}

// Initial fetch
fetchState();
