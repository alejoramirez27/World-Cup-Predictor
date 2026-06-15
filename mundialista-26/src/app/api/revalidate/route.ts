import { revalidatePath } from "next/cache";
import { NextResponse } from "next/server";

// Revalidación on-demand: sync_supabase.py llama a este endpoint al terminar,
// así la web se actualiza al instante sin esperar la ISR ni redeploy.
// Protegido con un token compartido (env REVALIDATE_TOKEN).
export async function POST(req: Request) {
  const token = req.headers.get("x-revalidate-token");
  const expected = process.env.REVALIDATE_TOKEN;
  if (!expected || token !== expected) {
    return NextResponse.json({ ok: false, error: "unauthorized" }, { status: 401 });
  }
  revalidatePath("/");
  revalidatePath("/tracking");
  revalidatePath("/partido/[id]", "page");
  return NextResponse.json({ ok: true, at: new Date().toISOString() });
}
