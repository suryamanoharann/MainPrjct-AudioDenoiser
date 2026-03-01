// --- 1. State & Constants ---
const THRESHOLD_VALUE = 0.005; // Keeps weak stems visible while model trains
let audioCtx;
let masterGain;
let presenceFilter;
let isPlaying = false;
let stemNodes = {};    
let audioBuffers = {}; 

const uploadZone = document.getElementById('upload-zone');
const mixerWorkspace = document.getElementById('mixer-workspace');
const stemBoard = document.getElementById('stem-board');
const playbackController = document.getElementById('playback-controller');
const playBtn = document.getElementById('play-pause-btn');
const masterVolumeSlider = document.getElementById('master-volume');

// --- 2. Web Audio API Initialization ---
function initAudioEngine() {
    if (audioCtx) return; // Prevent double init
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 0.8;

    presenceFilter = audioCtx.createBiquadFilter();
    presenceFilter.type = 'highshelf';
    presenceFilter.frequency.value = 3000; 
    presenceFilter.gain.value = 0;         

    presenceFilter.connect(masterGain);
    masterGain.connect(audioCtx.destination);
}

// --- 3. File Upload Logic (Drag & Drop) ---
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    initAudioEngine();
    
    const file = e.dataTransfer.files;
    if (!file) return;

    uploadZone.innerHTML = `<h2>Processing...</h2><p>Running audio through PyTorch model. This may take a minute...</p>`;

    const formData = new FormData();
    formData.append('audio_file', file);

    try {
        const response = await fetch('/process-audio', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error("Server processing failed.");
        
        const data = await response.json();
        await loadAndBuildUI(data.stems);
        
    } catch (error) {
        alert("Error connecting to backend: " + error.message);
        uploadZone.innerHTML = `<h2>Drag & Drop Audio File</h2><p>Try again</p>`;
    }
});

// --- 4. Loading & UI Building ---
async function loadAndBuildUI(stems) {
    const activeStems = stems.filter(stem => stem.rmsEnergy >= THRESHOLD_VALUE);

    if (activeStems.length === 0) {
        alert("Audio was too quiet or model output failed. Try another file.");
        uploadZone.innerHTML = `<h2>Drag & Drop Audio File</h2><p>Try again</p>`;
        return;
    }

    uploadZone.innerHTML = `<h2>Loading Audio Buffers...</h2>`;
    stemBoard.innerHTML = ''; 

    // Fetch and decode all active stems
    await Promise.all(activeStems.map(async (stem) => {
        try {
            const res = await fetch(stem.url);
            const arrayBuffer = await res.arrayBuffer();
            audioBuffers[stem.id] = await audioCtx.decodeAudioData(arrayBuffer);
            
            // Build Routing
            const gainNode = audioCtx.createGain();
            gainNode.gain.value = 0.8;
            gainNode.connect(presenceFilter); 
            stemNodes[stem.id] = { gainNode: gainNode, source: null };

            // Build UI
            const channelDiv = document.createElement('div');
            channelDiv.className = 'stem-channel';
            channelDiv.innerHTML = `
                <input type="range" class="vertical-slider" orient="vertical" min="0" max="1.5" step="0.01" value="0.8">
                <h4>${stem.name}</h4>
            `;

            channelDiv.querySelector('input').addEventListener('input', (e) => {
                stemNodes[stem.id].gainNode.gain.setTargetAtTime(parseFloat(e.target.value), audioCtx.currentTime, 0.01);
            });

            stemBoard.appendChild(channelDiv);
        } catch (err) {
            console.error(`Failed to load ${stem.name}`, err);
        }
    }));

    uploadZone.classList.add('hidden');
    mixerWorkspace.classList.remove('hidden');
    playbackController.classList.remove('hidden');
}

// --- 5. The 7th Control (Dial Logic) ---
const dial = document.getElementById('presence-dial');
const dialValueText = document.getElementById('dial-value');
let isDragging = false, startY, startVal = 0, currentDialRotation = 0; 

document.getElementById('presence-dial-container').addEventListener('mousedown', (e) => {
    isDragging = true; 
    startY = e.clientY; 
    startVal = currentDialRotation;
    document.body.style.cursor = 'ns-resize';
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    // Calculate rotation limits (-150 to 150 degrees)
    currentDialRotation = Math.max(-150, Math.min(150, startVal + ((startY - e.clientY) * 1.5)));
    dial.style.transform = `rotate(${currentDialRotation}deg)`;
    
    // Map rotation to EQ Gain (-15dB to +15dB)
    const eqGain = (currentDialRotation / 150) * 15;
    dialValueText.innerText = `${eqGain > 0 ? '+' : ''}${eqGain.toFixed(1)} dB`;
    
    if (presenceFilter) {
        presenceFilter.gain.setTargetAtTime(eqGain, audioCtx.currentTime, 0.01);
    }
});

window.addEventListener('mouseup', () => { 
    isDragging = false; 
    document.body.style.cursor = 'default';
});

// Master Volume Listener
masterVolumeSlider.addEventListener('input', (e) => {
    if (masterGain) {
        masterGain.gain.setTargetAtTime(parseFloat(e.target.value), audioCtx.currentTime, 0.01);
    }
});

// --- 6. Playback Control (Sample Accurate Sync) ---
playBtn.addEventListener('click', () => {
    if (!audioCtx) return;
    if (audioCtx.state === 'suspended') audioCtx.resume();

    if (!isPlaying) {
        // Start all stems precisely 50ms in the future to avoid phase cancellation
        const exactStartTime = audioCtx.currentTime + 0.05;
        
        Object.keys(stemNodes).forEach(stemId => {
            if (audioBuffers[stemId]) {
                const source = audioCtx.createBufferSource();
                source.buffer = audioBuffers[stemId];
                source.connect(stemNodes[stemId].gainNode);
                source.start(exactStartTime);
                stemNodes[stemId].source = source; 
            }
        });
        
        playBtn.innerText = 'Stop';
        isPlaying = true;
    } else {
        Object.values(stemNodes).forEach(node => {
            if(node.source) { 
                node.source.stop(); 
                node.source.disconnect(); 
            }
        });
        playBtn.innerText = 'Play';
        isPlaying = false;
    }
});