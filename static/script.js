let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let progressTimer = null;
let progressStep = 0;

const AGENT_STEPS = [
    "Finding flights…",
    "Searching hotels…",
    "Checking weather…",
    "Building your itinerary…",
    "Writing the final plan…"
];

function setPrompt(button, text) {
    document.getElementById("userInput").value = text;
    document.querySelectorAll(".quick-prompts button").forEach((btn) => {
        btn.classList.toggle("active", btn === button);
    });
}

function setLoading(isLoading) {
    const sendBtn = document.getElementById("sendBtn");
    const btnText = document.getElementById("btnText");
    const btnLoader = document.getElementById("btnLoader");

    sendBtn.disabled = isLoading;

    if (isLoading) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
        startAgentProgress();
    } else {
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
        stopAgentProgress();
    }
}

function renderProgress() {
    const status = document.getElementById("progressStatus");
    const items = document.querySelectorAll(".progress-steps li");

    if (status) {
        status.textContent = AGENT_STEPS[progressStep];
    }

    items.forEach((item, index) => {
        item.classList.toggle("active", index === progressStep);
        item.classList.toggle("done", index < progressStep);
    });
}

function startAgentProgress() {
    const panel = document.getElementById("agentProgress");

    progressStep = 0;
    panel.classList.remove("hidden");
    renderProgress();

    clearInterval(progressTimer);
    progressTimer = setInterval(() => {
        if (progressStep < AGENT_STEPS.length - 1) {
            progressStep += 1;
            renderProgress();
        }
    }, 2400);
}

function stopAgentProgress(complete = false) {
    const panel = document.getElementById("agentProgress");

    clearInterval(progressTimer);
    progressTimer = null;

    if (complete) {
        progressStep = AGENT_STEPS.length;
        document.getElementById("progressStatus").textContent = "Plan ready.";
        document.querySelectorAll(".progress-steps li").forEach((item) => {
            item.classList.remove("active");
            item.classList.add("done");
        });
        setTimeout(() => panel.classList.add("hidden"), 900);
    } else {
        panel.classList.add("hidden");
    }
}

function showError(message) {
    const errorBox = document.getElementById("errorBox");

    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    const errorBox = document.getElementById("errorBox");

    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

function showResult(answer, threadId, llmCalls) {
    latestAnswerMarkdown = answer;

    const resultSection = document.getElementById("resultSection");
    const resultBox = document.getElementById("resultBox");
    const threadInfo = document.getElementById("threadInfo");
    const planMeta = document.getElementById("planMeta");

    if (typeof marked !== "undefined") {
        resultBox.innerHTML = marked.parse(answer);
    } else {
        resultBox.innerText = answer;
    }

    const callLabel = llmCalls === 1 ? "1 agent call" : `${llmCalls || 0} agent calls`;
    planMeta.textContent = `4 specialists · ${callLabel}`;
    threadInfo.textContent = `Thread ID: ${threadId}`;

    resultSection.classList.remove("hidden");
    resultSection.classList.remove("rise-in");
    void resultSection.offsetWidth;
    resultSection.classList.add("rise-in");

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

async function sendMessage() {
    hideError();

    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) {
        showError("Please enter your travel request first.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/travel", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                thread_id: currentThreadId
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong.");
        }

        currentThreadId = data.thread_id;
        localStorage.setItem("travel_thread_id", currentThreadId);

        stopAgentProgress(true);
        showResult(data.answer, data.thread_id, data.llm_calls);

    } catch (error) {
        stopAgentProgress(false);
        showError(error.message);
    } finally {
        const sendBtn = document.getElementById("sendBtn");
        const btnText = document.getElementById("btnText");
        const btnLoader = document.getElementById("btnLoader");

        sendBtn.disabled = false;
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}

function copyResult() {
    const resultBox = document.getElementById("resultBox");
    const text = resultBox.innerText;

    if (!text) {
        return;
    }

    navigator.clipboard.writeText(text)
        .then(() => {
            const copyBtn = document.querySelector(".copy-btn");
            const oldText = copyBtn.textContent;

            copyBtn.textContent = "Copied!";

            setTimeout(() => {
                copyBtn.textContent = oldText;
            }, 1400);
        })
        .catch(() => {
            showError("Could not copy result.");
        });
}

function downloadPDF() {
    const pdfContent = document.getElementById("pdfContent");

    if (!latestAnswerMarkdown || !pdfContent) {
        showError("No travel plan available to download.");
        return;
    }

    const downloadBtn = document.querySelector(".download-btn");
    const oldText = downloadBtn.textContent;

    downloadBtn.textContent = "Preparing PDF...";
    downloadBtn.disabled = true;

    const options = {
        margin: 0.5,
        filename: "ai-travel-plan.pdf",
        image: {
            type: "jpeg",
            quality: 0.98
        },
        html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#fffaf3"
        },
        jsPDF: {
            unit: "in",
            format: "a4",
            orientation: "portrait"
        },
        pagebreak: {
            mode: ["avoid-all", "css", "legacy"]
        }
    };

    html2pdf()
        .set(options)
        .from(pdfContent)
        .save()
        .then(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
        })
        .catch(() => {
            downloadBtn.textContent = oldText;
            downloadBtn.disabled = false;
            showError("Could not download PDF.");
        });
}

document.addEventListener("keydown", function(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        sendMessage();
    }
});

document.addEventListener("DOMContentLoaded", function() {
    const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
    const modKey = document.getElementById("modKey");

    if (modKey && isMac) {
        modKey.textContent = "⌘";
    }
});
