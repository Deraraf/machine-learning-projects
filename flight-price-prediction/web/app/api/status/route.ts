import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  const projectRoot = path.resolve(process.cwd(), "..");
  const reportPath = path.join(projectRoot, "outputs", "reports", "metrics.json");
  const modelPath = path.join(projectRoot, "outputs", "models", "flight_price_model.joblib");

  if (!fs.existsSync(reportPath) || !fs.existsSync(modelPath)) {
    return NextResponse.json(
      { error: "Train the model and generate outputs before using the demo." },
      { status: 503 },
    );
  }

  const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
  return NextResponse.json({
    selectedModel: report.selected_model,
    testMetrics: report.test_metrics,
    cleaning: report.cleaning,
  });
}
