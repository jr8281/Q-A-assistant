const $ = (id) => document.getElementById(id);
const form = $('searchForm'), input = $('queryInput'), topK = $('topK');
const submitBtn = $('submitBtn'), resultsDiv = $('results'), answerBox = $('answerBox');
const sourcesBox = $('sourcesBox'), countEl = $('count'), metaEl = $('meta'), errorBox = $('error');

const NOT_FOUND_TEXT = 'was not found in the documents';
const EXAMPLES = [
    'What are the four core functions of the AI RMF?',
    'What risks are unique to generative AI?',
    'What are the functions of the Cybersecurity Framework 2.0?',
];

EXAMPLES.forEach(q => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = q;
    chip.addEventListener('click', () => { input.value = q; form.requestSubmit(); });
    $('examples').appendChild(chip);
});

// Show what is indexed, straight from the API
fetch('/health').then(r => r.json()).then(h => {
    $('corpus').textContent = `${h.documents.length} PDFs · ${h.chunks} chunks indexed · LLM: ${h.llm_model}`;
}).catch(() => {});

// Enter submits, Shift+Enter makes a new line
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

// Minimal markdown: **bold**, bullet lists, [file.pdf, p.N] citation chips.
// All text is HTML-escaped FIRST, so injecting the result with innerHTML is safe.
function renderMarkdown(raw) {
    const esc = raw.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const inline = (s) => s
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[([^\]]*\.pdf[^\]]*)\]/g, '<span class="cite">$1</span>');
    const out = [];
    let inList = false;
    for (const line of esc.split('\n')) {
        const item = line.match(/^\s*(?:[*\-•]|\d+\.)\s+(.*)/);
        if (item) {
            if (!inList) { out.push('<ul>'); inList = true; }
            out.push(`<li>${inline(item[1])}</li>`);
        } else {
            if (inList) { out.push('</ul>'); inList = false; }
            if (line.trim()) out.push(`<p>${inline(line)}</p>`);
        }
    }
    if (inList) out.push('</ul>');
    return out.join('');
}

function renderSources(sources) {
    sourcesBox.replaceChildren();
    sources.forEach((src, index) => {
        const card = document.createElement('article');
        card.className = 'card';

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = `#${index + 1} MATCH`;

        const title = document.createElement('h3');
        title.textContent = src.source;      // textContent prevents HTML injection

        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = `Page ${src.page} · similarity ${src.similarity.toFixed(2)} · score ${src.score.toFixed(2)}`;

        const divider = document.createElement('hr');
        divider.className = 'divider';

        const excerptLabel = document.createElement('div');
        excerptLabel.className = 'label';
        excerptLabel.textContent = 'EXCERPT';

        const text = document.createElement('p');
        text.textContent = src.text;

        card.append(label, title, badge, divider, excerptLabel, text);
        sourcesBox.appendChild(card);
    });
}

// Reads the Server-Sent Events stream and calls onEvent(eventName, data) for each message
async function streamQuery(payload, onEvent) {
    const res = await fetch('/api/v1/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid request.');
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const messages = buffer.split('\n\n');
        buffer = messages.pop();
        for (const msg of messages) {
            const event = /^event: (.+)$/m.exec(msg)?.[1];
            const data = /^data: (.+)$/m.exec(msg)?.[1];
            if (event && data) onEvent(event, JSON.parse(data));
        }
    }
}

function setBusy(busy) {
    submitBtn.disabled = busy;
    submitBtn.textContent = busy ? 'Thinking...' : 'Ask';
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setBusy(true);
    errorBox.hidden = true;
    answerBox.replaceChildren();
    sourcesBox.replaceChildren();
    metaEl.textContent = '';
    resultsDiv.hidden = false;

    let answer = '';
    try {
        await streamQuery(
            { question: input.value.trim(), top_k: Number(topK.value) },
            (event, data) => {
                if (event === 'sources') {
                    renderSources(data.sources);
                    countEl.textContent = data.sources.length
                        ? `${data.sources.length} sources found` : 'No relevant sources found';
                } else if (event === 'token') {
                    answer += data.text;
                    answerBox.innerHTML = renderMarkdown(answer);
                } else if (event === 'error') {
                    throw new Error(data.detail);
                } else if (event === 'done') {
                    if (answer.toLowerCase().includes(NOT_FOUND_TEXT)) {
                        sourcesBox.replaceChildren();
                        countEl.textContent = 'No relevant sources found';
                    }
                    metaEl.textContent = `${data.mode} search · retrieval ${data.retrieval_ms} ms · ` +
                        `generation ${data.generation_ms} ms · total ${data.latency_ms} ms · ${data.model}`;
                }
            },
        );
    } catch (err) {
        resultsDiv.hidden = true;
        errorBox.textContent = err.message || 'Something went wrong. Please try again.';
        errorBox.hidden = false;
    } finally {
        setBusy(false);
    }
});