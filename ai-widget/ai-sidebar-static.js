(() => {
  if (typeof window !== "undefined" && window.__HTLB_AI_STATIC_MOUNTED__) return;
  if (typeof window !== "undefined") window.__HTLB_AI_STATIC_MOUNTED__ = true;

  const STORAGE_KEY = "htlb_ai_static_config_v1";
  const DEFAULT_CONFIG = {
    api_key: "",
    base_url: "https://api.deepseek.com/v1",
    model: "deepseek-chat",
    top_k: 5,
  };

  const SAMPLE_QUESTIONS = [
    "低钠盐怎么选？哪些人不能吃？",
    "怎样低成本改善睡眠质量？",
    "普通人买保险最容易踩什么坑？",
    "体检有哪些高性价比的必查项目？",
  ];

  const STOPWORDS = new Set([
    "的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "或", "一个", "没有",
    "我们", "你们", "他们", "自己", "什么", "怎么", "如何", "为什么", "可以", "应该",
    "需要", "不要", "有没有", "哪些", "请问", "一下", "帮我", "告诉", "关于", "这个",
    "那个", "如果", "但是", "因为", "所以", "还是", "以及", "或者", "并且", "已经",
    "非常", "比较", "一些", "很多", "最", "更", "很", "在", "把", "被", "对", "给",
    "从", "向", "到", "为", "以", "上", "下", "中", "里", "吗", "呢", "吧", "啊",
    "the", "a", "an", "is", "are", "to", "of", "in", "on", "for", "with", "and", "or",
  ]);

  const SYSTEM_PROMPT = `你是《高性价比人生指南》的本地知识库问答助手。
请严格遵守以下规则作答：
1. 【严格基于原文】你只能使用下方「召回原文」中明确出现的信息来回答用户问题，严禁引入外部记忆、常识推测或原文未提及的建议。
2. 【严禁编造数字】原文中的任何数字、金额、比例、时间、剂量、证据等级必须与原文完全一致，不得估算、改写或编造任何数字。
3. 【强制标注出处】每一个要点或结论后面，都必须明确标注对应的章节和条号，格式如「（第1节第5条）」。
4. 【如实说明缺失】如果召回的原文中没有足够信息回答用户问题，请直接说明「当前召回的原文中未包含该问题的直接依据」，并仅列出最接近的原文要点，切勿强行作答。
5. 【表达风格】回答简洁清晰、分点列出，优先用“说人话”的方式忠实转述原文建议，并给出原文提到的成本与收益。`;

  function tokenize(text) {
    if (!text) return [];
    const lower = String(text).toLowerCase();
    const tokens = [];

    const alnumMatches = lower.match(/[a-z0-9]+(?:\.[0-9]+)?/g);
    if (alnumMatches) {
      for (const w of alnumMatches) {
        if (!STOPWORDS.has(w)) tokens.push(w);
      }
    }

    const cjkSegments = lower.match(/[\u4e00-\u9fff]+/g);
    if (cjkSegments) {
      for (const seg of cjkSegments) {
        if (seg.length === 1) {
          if (!STOPWORDS.has(seg)) tokens.push(seg);
        } else {
          for (let i = 0; i < seg.length; i++) {
            const ch = seg[i];
            if (!STOPWORDS.has(ch)) tokens.push(ch);
            if (i < seg.length - 1) {
              const bg = seg.slice(i, i + 2);
              if (!STOPWORDS.has(bg)) tokens.push(bg);
            }
          }
        }
      }
    }
    return tokens;
  }

  class BM25Index {
    constructor(items, k1 = 1.5, b = 0.75) {
      this.items = items || [];
      this.k1 = k1;
      this.b = b;
      this.nDocs = this.items.length;
      this.docFreqs = [];
      this.docLens = [];
      this.df = new Map();
      this.avgdl = 1.0;
      this._build();
    }

    _build() {
      let totalLen = 0;
      for (const item of this.items) {
        const tf = new Map();
        const addTokens = (arr, weight) => {
          for (const t of arr) {
            tf.set(t, (tf.get(t) || 0) + weight);
          }
        };
        addTokens(tokenize(item.title), 3);
        addTokens(tokenize(item.plain_text), 2);
        addTokens(tokenize(item.chapter_title), 2);
        addTokens(
          tokenize(`${item.cost || ""} ${item.benefit || ""} ${item.note || ""} ${item.source || ""}`),
          1
        );

        let dl = 0;
        for (const [term, count] of tf.entries()) {
          dl += count;
          this.df.set(term, (this.df.get(term) || 0) + 1);
        }
        dl = dl || 1;
        this.docFreqs.push(tf);
        this.docLens.push(dl);
        totalLen += dl;
      }
      this.avgdl = this.nDocs ? totalLen / this.nDocs : 1.0;
    }

    search(query, topK = 5) {
      const qClean = String(query || "").trim().toLowerCase();
      if (!qClean || !this.nDocs) return [];
      const qTokens = tokenize(qClean);
      if (!qTokens.length) return [];

      const scored = [];
      for (let idx = 0; idx < this.nDocs; idx++) {
        const item = this.items[idx];
        const tf = this.docFreqs[idx];
        const dl = this.docLens[idx];
        let score = 0;

        for (const term of qTokens) {
          const freq = tf.get(term) || 0;
          if (!freq) continue;
          const dfT = this.df.get(term) || 0;
          const idf = Math.log(1 + (this.nDocs - dfT + 0.5) / (dfT + 0.5));
          const denom = freq + this.k1 * (1 - this.b + (this.b * dl) / this.avgdl);
          score += idf * ((freq * (this.k1 + 1)) / denom);
        }

        if (qClean.length >= 2) {
          if ((item.title || "").toLowerCase().includes(qClean)) {
            score += 6.0;
          } else if ((item.raw_markdown || "").toLowerCase().includes(qClean)) {
            score += 2.5;
          }
        }
        if (
          (item.citation || "").toLowerCase().includes(qClean) ||
          qClean.includes(`第${item.chapter_num}节第${item.item_num}条`)
        ) {
          score += 12.0;
        }

        if (score > 0) {
          scored.push({ item, score: Number(score.toFixed(4)) });
        }
      }

      scored.sort((a, b) => b.score - a.score);
      return scored.slice(0, topK).map((x) => ({ ...x.item, score: x.score }));
    }
  }

  // 允许 Node.js 测试脚本直接导入核心检索类
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { tokenize, BM25Index };
    return;
  }

  function loadConfig() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
      }
    } catch (_) {}
    return { ...DEFAULT_CONFIG };
  }

  function saveConfig(cfg) {
    const merged = {
      ...loadConfig(),
      ...cfg,
      base_url: String(cfg.base_url || "").trim().replace(/\/+$/, ""),
      model: String(cfg.model || "deepseek-chat").trim() || "deepseek-chat",
      top_k: Math.max(1, Math.min(10, Number(cfg.top_k) || 5)),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
    return merged;
  }

  function maskKey(k) {
    if (!k) return "";
    if (k.length <= 8) return "*".repeat(k.length);
    return `${k.slice(0, 4)}${"*".repeat(k.length - 8)}${k.slice(-4)}`;
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderSimpleMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    return html;
  }

  function parseMarkdownChapter(text, fileName) {
    const mFile = fileName.replace(/\.md$/i, "").match(/^0*(\d+)(?:[-_\s\.]+(.+))?$/);
    let chapterNum = mFile ? mFile[1] : fileName.replace(/\.md$/i, "");
    let chapterTitle = mFile && mFile[2] ? mFile[2].trim() : chapterNum;

    const lines = text.split(/\r?\n/);
    for (const rawLine of lines.slice(0, 20)) {
      const s = rawLine.trim();
      if (s.startsWith("# ") || s.startsWith("## ")) {
        const heading = s.replace(/^#+\s*/, "").trim();
        const mh = heading.match(/^(?:第\s*0*(\d+)\s*[节章][\s：:、\.\-]*|0*(\d+)[\s\.、：:\-]+)(.+)$/);
        if (mh) {
          chapterNum = chapterNum || mh[1] || mh[2] || "";
          chapterTitle = mh[3].trim();
        } else {
          chapterTitle = heading;
        }
        break;
      }
    }

    const items = [];
    const regex = /^###\s+(.+)$/gm;
    const matches = [];
    let m;
    while ((m = regex.exec(text)) !== null) {
      matches.push({ heading: m[1].trim(), index: m.index, end: regex.lastIndex });
    }

    for (let i = 0; i < matches.length; i++) {
      const cur = matches[i];
      const nextIdx = i + 1 < matches.length ? matches[i + 1].index : text.length;
      const body = text.slice(cur.end, nextIdx).trim();
      const mi = cur.heading.match(/^(?:第\s*)?(\d+)\s*(?:条|[\.、:：\)\s]+)\s*(.+)$/);
      const itemNum = mi ? mi[1] : String(i + 1);
      const title = mi ? mi[2].trim() : cur.heading;

      const fields = { cost: "", plain_text: "", benefit: "", grade: "", source: "", note: "" };
      const map = { 成本: "cost", 说人话: "plain_text", 收益: "benefit", 证据等级: "grade", 来源: "source", 备注: "note" };
      let currentKey = null;
      for (const rLine of body.split(/\r?\n/)) {
        const line = rLine.trim();
        if (!line) continue;
        let hit = false;
        for (const [cn, en] of Object.entries(map)) {
          const fm = line.match(new RegExp(`^(?:[-*•]\\s*)?(?:\\*\\*)?${cn}(?:\\*\\*)?\\s*[：:]\\s*(.*)$`));
          if (fm) {
            fields[en] = fm[1].trim();
            currentKey = en;
            hit = true;
            break;
          }
        }
        if (!hit && currentKey && !line.startsWith("#")) {
          fields[currentKey] += (fields[currentKey] ? "\n" : "") + line;
        }
      }
      if (fields.grade) {
        const mg = fields.grade.match(/^([ABCabc])/);
        if (mg) fields.grade = mg[1].toUpperCase();
      }

      items.push({
        id: `${chapterNum}-${itemNum}`,
        chapter_num: chapterNum,
        chapter_title: chapterTitle,
        item_num: itemNum,
        title,
        citation: `第${chapterNum}节第${itemNum}条（${chapterTitle}）`,
        grade: fields.grade,
        cost: fields.cost,
        plain_text: fields.plain_text,
        benefit: fields.benefit,
        source: fields.source,
        note: fields.note,
        raw_markdown: `### ${itemNum}. ${title}\n${body}`.trim(),
        file_name: fileName,
      });
    }
    return items;
  }

  async function loadBookIndex() {
    // 优先加载预构建的 ai-widget/book-index.json（单次请求，手机秒开）
    try {
      const res = await fetch("ai-widget/book-index.json");
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.items) && data.items.length > 0) {
          return {
            items: data.items,
            chaptersCount: data.chapters_count || 0,
          };
        }
      }
    } catch (_) {}

    // 兜底：从 README.md 解析 book/*.md 路径并并行加载
    const readmeRes = await fetch("README.md");
    if (!readmeRes.ok) throw new Error("无法读取 book-index.json 或 README.md");
    const readme = await readmeRes.text();
    const files = Array.from(new Set(readme.match(/book\/[^\s\)\]]+\.md/g) || []));
    const chapters = await Promise.all(
      files.map(async (relPath) => {
        try {
          const r = await fetch(relPath);
          if (!r.ok) return [];
          const txt = await r.text();
          return parseMarkdownChapter(txt, relPath.split("/").pop());
        } catch (_) {
          return [];
        }
      })
    );
    const allItems = chapters.flat();
    return { items: allItems, chaptersCount: files.length };
  }

  async function callLLMApi(cfg, query, retrieved) {
    if (!cfg.api_key) {
      throw new Error("尚未配置 API Key。请点击右上角「⚙️ 配置」填入你的 API Key、Base URL 和模型名称（密钥仅保存在当前手机浏览器本地）。");
    }
    if (!cfg.base_url) {
      throw new Error("尚未配置 Base URL。");
    }

    const contextBlocks = retrieved.map(
      (item, idx) =>
        `【参考原文 ${idx + 1}】${item.citation} | 证据等级：${item.grade || "未标注"}\n${item.raw_markdown}`
    );
    const userPrompt =
      `用户问题：${query}\n\n` +
      `以下是从《高性价比人生指南》book/ 目录中检索出的最相关 ${retrieved.length} 条原文：\n\n` +
      contextBlocks.join("\n\n---\n\n") +
      `\n\n请严格依据以上原文回答用户问题，并附上对应章节和条号。`;

    const base = cfg.base_url.replace(/\/+$/, "");
    const endpoint = base.endsWith("/chat/completions") ? base : `${base}/chat/completions`;

    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${cfg.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: cfg.model || "deepseek-chat",
        temperature: 0.2,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
      }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`大模型 API 返回错误 (HTTP ${res.status})：${errText.slice(0, 260)}`);
    }
    const data = await res.json();
    return String(data?.choices?.[0]?.message?.content || "").trim();
  }

  function mountUI() {
    let bm25 = new BM25Index([]);
    let currentMode = "chat";
    let isLoading = false;

    const fab = document.createElement("button");
    fab.className = "htlb-ai-fab";
    fab.type = "button";
    fab.innerHTML = "<span>✨</span><span>AI 语义问答</span>";

    const backdrop = document.createElement("div");
    backdrop.className = "htlb-ai-backdrop";

    const drawer = document.createElement("aside");
    drawer.className = "htlb-ai-drawer";
    drawer.setAttribute("aria-label", "AI 语义检索与问答");
    drawer.innerHTML = `
      <div class="htlb-ai-header">
        <div class="htlb-ai-title-wrap">
          <h3 class="htlb-ai-title">AI 语义检索 / 问答</h3>
          <div class="htlb-ai-subtitle" id="htlb-ai-status-text">正在浏览器本地建立索引...</div>
        </div>
        <div class="htlb-ai-header-actions">
          <button type="button" class="htlb-ai-icon-btn" id="htlb-ai-settings-toggle">⚙️ 配置</button>
          <button type="button" class="htlb-ai-icon-btn" id="htlb-ai-close">✕</button>
        </div>
      </div>

      <div class="htlb-ai-settings" id="htlb-ai-settings-panel">
        <div class="htlb-ai-field">
          <label for="htlb-ai-base-url">API Base URL（OpenAI 兼容接口）</label>
          <input id="htlb-ai-base-url" type="url" placeholder="https://api.deepseek.com/v1" />
        </div>
        <div class="htlb-ai-field">
          <label for="htlb-ai-api-key">API Key（仅存在本机浏览器，不上传服务器）</label>
          <input id="htlb-ai-api-key" type="password" placeholder="sk-..." />
        </div>
        <div class="htlb-ai-settings-row">
          <div class="htlb-ai-field">
            <label for="htlb-ai-model">模型名称</label>
            <input id="htlb-ai-model" type="text" placeholder="deepseek-chat / gpt-4o-mini" />
          </div>
          <div class="htlb-ai-field" style="max-width: 110px;">
            <label for="htlb-ai-topk">召回条数</label>
            <select id="htlb-ai-topk">
              <option value="3">3 条</option>
              <option value="4">4 条</option>
              <option value="5" selected>5 条</option>
            </select>
          </div>
        </div>
        <div class="htlb-ai-settings-footer">
          <span id="htlb-ai-config-hint" style="color:#64748b;font-size:12px;"></span>
          <button type="button" class="htlb-ai-save-btn" id="htlb-ai-save-config">保存配置</button>
        </div>
      </div>

      <div class="htlb-ai-mode-bar">
        <button type="button" class="htlb-ai-mode-btn active" data-mode="chat">AI 问答（召回 + 生成）</button>
        <button type="button" class="htlb-ai-mode-btn" data-mode="search">仅本地 BM25 检索</button>
      </div>

      <div class="htlb-ai-body" id="htlb-ai-body"></div>

      <div class="htlb-ai-footer">
        <div class="htlb-ai-input-wrap">
          <textarea
            id="htlb-ai-input"
            class="htlb-ai-textarea"
            rows="2"
            placeholder="输入自然语言问题..."
          ></textarea>
          <button type="button" class="htlb-ai-send-btn" id="htlb-ai-send">发送</button>
        </div>
      </div>
    `;

    document.body.appendChild(fab);
    document.body.appendChild(backdrop);
    document.body.appendChild(drawer);

    const statusText = drawer.querySelector("#htlb-ai-status-text");
    const settingsPanel = drawer.querySelector("#htlb-ai-settings-panel");
    const settingsToggle = drawer.querySelector("#htlb-ai-settings-toggle");
    const closeBtn = drawer.querySelector("#htlb-ai-close");
    const bodyEl = drawer.querySelector("#htlb-ai-body");
    const inputEl = drawer.querySelector("#htlb-ai-input");
    const sendBtn = drawer.querySelector("#htlb-ai-send");
    const baseUrlEl = drawer.querySelector("#htlb-ai-base-url");
    const apiKeyEl = drawer.querySelector("#htlb-ai-api-key");
    const modelEl = drawer.querySelector("#htlb-ai-model");
    const topKEl = drawer.querySelector("#htlb-ai-topk");
    const configHintEl = drawer.querySelector("#htlb-ai-config-hint");
    const saveConfigBtn = drawer.querySelector("#htlb-ai-save-config");
    const modeBtns = drawer.querySelectorAll(".htlb-ai-mode-btn");

    let chaptersCount = 0;

    function syncConfigUI() {
      const cfg = loadConfig();
      baseUrlEl.value = cfg.base_url || "";
      modelEl.value = cfg.model || "";
      topKEl.value = String(cfg.top_k || 5);
      configHintEl.textContent = cfg.api_key
        ? `已保存密钥：${maskKey(cfg.api_key)}`
        : "尚未填写 API Key";
      if (bm25.nDocs > 0) {
        const apiTag = cfg.api_key ? `模型：${cfg.model}` : "未配密钥（可用本地检索）";
        statusText.textContent = `已索引 ${chaptersCount} 节 · ${bm25.nDocs} 条 | ${apiTag}`;
      }
    }

    function renderWelcome() {
      bodyEl.innerHTML = `
        <div class="htlb-ai-empty">
          <div>纯前端免后端版：索引直接运行在你的手机浏览器内，无需开电脑。</div>
          <ul style="margin:6px 0 6px 18px;padding:0;">
            <li><strong>AI 问答</strong>：召回最相关 3–5 条原文，严格基于原文作答并标注章节条号。</li>
            <li><strong>仅本地检索</strong>：无需 API Key，离线秒查原文。</li>
          </ul>
          <div class="htlb-ai-samples">
            ${SAMPLE_QUESTIONS.map(
              (q) => `<button type="button" class="htlb-ai-sample-chip">${escapeHtml(q)}</button>`
            ).join("")}
          </div>
        </div>
      `;
      bodyEl.querySelectorAll(".htlb-ai-sample-chip").forEach((btn) => {
        btn.addEventListener("click", () => {
          inputEl.value = btn.textContent || "";
          handleSubmit();
        });
      });
    }

    function renderCitations(citations) {
      if (!citations || !citations.length) {
        return '<div class="htlb-ai-empty">未检索到相关原文条目，请换个关键词试试。</div>';
      }
      return `
        <div class="htlb-ai-section-label">召回原文（共 ${citations.length} 条）</div>
        ${citations
          .map((item) => {
            const gradeClass = item.grade ? `htlb-ai-grade-${escapeHtml(item.grade)}` : "";
            const gradeBadge = item.grade
              ? `<span class="htlb-ai-grade ${gradeClass}">证据 ${escapeHtml(item.grade)} 级</span>`
              : "";
            const plain = item.plain_text
              ? `<div class="htlb-ai-card-plain">${escapeHtml(item.plain_text)}</div>`
              : "";
            return `
              <div class="htlb-ai-card">
                <div class="htlb-ai-card-head">
                  <span class="htlb-ai-citation">${escapeHtml(item.citation)}</span>
                  ${gradeBadge}
                </div>
                <div class="htlb-ai-card-title">${escapeHtml(item.title)}</div>
                ${plain}
                <details>
                  <summary>查看完整原文</summary>
                  <div class="htlb-ai-raw">${escapeHtml(item.raw_markdown)}</div>
                </details>
              </div>
            `;
          })
          .join("")}
      `;
    }

    async function handleSubmit() {
      const query = inputEl.value.trim();
      if (!query || isLoading) return;
      if (!bm25.nDocs) {
        bodyEl.innerHTML = '<div class="htlb-ai-error">索引尚未加载完成，请稍候再试。</div>';
        return;
      }

      isLoading = true;
      sendBtn.disabled = true;
      const cfg = loadConfig();
      const topK = Number(topKEl.value) || cfg.top_k || 5;
      const retrieved = bm25.search(query, topK);

      if (currentMode === "search") {
        bodyEl.innerHTML = renderCitations(retrieved);
        isLoading = false;
        sendBtn.disabled = false;
        return;
      }

      if (!retrieved.length) {
        bodyEl.innerHTML =
          '<div class="htlb-ai-empty">未在原文中检索到直接相关条目，建议换更具体的关键词再试。</div>';
        isLoading = false;
        sendBtn.disabled = false;
        return;
      }

      bodyEl.innerHTML = '<div class="htlb-ai-empty">已本地召回原文，正在调用大模型生成回答...</div>';
      try {
        const answer = await callLLMApi(cfg, query, retrieved);
        bodyEl.innerHTML = `
          <div class="htlb-ai-answer-box">
            <div class="htlb-ai-answer-meta">
              <span>严格基于原文生成</span>
              <span>${escapeHtml(cfg.model || "")}</span>
            </div>
            <div>${renderSimpleMarkdown(answer)}</div>
          </div>
          ${renderCitations(retrieved)}
        `;
      } catch (err) {
        const msg = err && err.message ? err.message : String(err);
        bodyEl.innerHTML = `
          <div class="htlb-ai-error">${escapeHtml(msg)}</div>
          <div class="htlb-ai-empty">提示：若出现跨域(CORS)或网络错误，可先切换到「仅本地 BM25 检索」模式查看原文，或更换支持浏览器直连的 API（如 DeepSeek、硅基流动、OpenRouter 等）。</div>
          ${renderCitations(retrieved)}
        `;
      } finally {
        isLoading = false;
        sendBtn.disabled = false;
      }
    }

    fab.addEventListener("click", () => {
      drawer.classList.add("open");
      backdrop.classList.add("open");
    });
    const closeDrawer = () => {
      drawer.classList.remove("open");
      backdrop.classList.remove("open");
    };
    backdrop.addEventListener("click", closeDrawer);
    closeBtn.addEventListener("click", closeDrawer);
    settingsToggle.addEventListener("click", () => settingsPanel.classList.toggle("open"));

    saveConfigBtn.addEventListener("click", () => {
      const next = {
        base_url: baseUrlEl.value.trim(),
        model: modelEl.value.trim(),
        top_k: Number(topKEl.value) || 5,
      };
      if (apiKeyEl.value.trim()) next.api_key = apiKeyEl.value.trim();
      saveConfig(next);
      apiKeyEl.value = "";
      syncConfigUI();
      configHintEl.textContent = "已保存到本机浏览器！";
    });

    sendBtn.addEventListener("click", handleSubmit);
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    });
    modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        currentMode = btn.dataset.mode || "chat";
        modeBtns.forEach((b) => b.classList.toggle("active", b === btn));
      });
    });

    renderWelcome();
    syncConfigUI();

    loadBookIndex()
      .then(({ items, chaptersCount: ch }) => {
        chaptersCount = ch;
        bm25 = new BM25Index(items);
        syncConfigUI();
      })
      .catch((err) => {
        statusText.textContent = `索引加载失败：${err.message}`;
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountUI);
  } else {
    mountUI();
  }
})();
