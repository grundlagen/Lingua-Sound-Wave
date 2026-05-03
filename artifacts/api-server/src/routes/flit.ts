import { Router, type IRouter } from "express";
import { runFlit } from "../lib/flit";

const router: IRouter = Router();

router.post("/flit/run", async (req, res) => {
  const body = req.body as {
    text?: unknown;
    language?: unknown;
    inputParaphrases?: unknown;
    targetRenderings?: unknown;
    topK?: unknown;
  };
  const text = typeof body.text === "string" ? body.text.trim() : "";
  const language = body.language === "en" || body.language === "fr" ? body.language : null;
  if (!text) {
    res.status(400).json({ error: "text is required" });
    return;
  }
  if (!language) {
    res.status(400).json({ error: "language must be 'en' or 'fr'" });
    return;
  }
  const inputParaphrases = typeof body.inputParaphrases === "number" ? Math.max(1, Math.min(8, Math.floor(body.inputParaphrases))) : 6;
  const targetRenderings = typeof body.targetRenderings === "number" ? Math.max(1, Math.min(6, Math.floor(body.targetRenderings))) : 4;
  const topK = typeof body.topK === "number" ? Math.max(1, Math.min(8, Math.floor(body.topK))) : 5;

  try {
    const result = await runFlit({ text, language, inputParaphrases, targetRenderings, topK });
    res.json(result);
  } catch (err) {
    req.log.error({ err }, "flit: run failed");
    res.status(502).json({ error: "Flit failed", detail: err instanceof Error ? err.message : String(err) });
  }
});

export default router;
