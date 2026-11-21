import { spawn } from "node:child_process";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const record = await request.json();
    const projectRoot = path.resolve(process.cwd(), "..");
    const python = path.join(projectRoot, ".venv", "bin", "python");
    const script = path.join(process.cwd(), "scripts", "predict.py");

    const result = await new Promise<{ prediction: number }>((resolve, reject) => {
      const child = spawn(python, [script], {
        cwd: projectRoot,
        env: {
          ...process.env,
          PYTHONPATH: path.join(projectRoot, "src"),
        },
      });

      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("error", reject);
      child.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(stderr || `Prediction process exited with code ${code}`));
          return;
        }
        try {
          resolve(JSON.parse(stdout.trim()));
        } catch {
          reject(new Error("The model returned an invalid response"));
        }
      });
      child.stdin.write(JSON.stringify(record));
      child.stdin.end();
    });

    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Prediction failed";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
