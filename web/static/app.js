const audioPlayer = document.getElementById("audioPlayer");
const currentSong = document.getElementById("currentSong");
const recommendationsDiv = document.getElementById("recommendations");


async function playSong(songName, songid) {
    currentSong.innerText = songName;

    audioPlayer.src = `/music/${encodeURIComponent(songid)}`;

    audioPlayer.play();

    loadRecommendations(songid);
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "\\'");
}

async function loadRecommendations(songid) {
    recommendationsDiv.innerHTML = "Loading recommendations...";

    const response = await fetch(
        `/api/recommend/${encodeURIComponent(songid)}`
    );

    const data = await response.json();

    recommendationsDiv.innerHTML = "";

    data.recommendations.forEach((song) => {
        const item = document.createElement("div");

        item.className = "recommendation-item";

        filename = escapeHtml(song.filename);

        item.innerHTML = `
            <div>
                <div class="recommendation-title">
                    ${song.filename}
                </div>

                <div class="recommendation-score">
                    Match Score: ${song.score}
                </div>
            </div>

            <button onclick="playSong('${filename}', '${song.id}')">
                Play
            </button>
        `;

        recommendationsDiv.appendChild(item);
    });
}
