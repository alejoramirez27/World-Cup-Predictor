import { ImageResponse } from "next/og";
import { getMatch } from "@/lib/queries";

export const alt = "Predicción del partido — mundialista·26";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// OG image dinámica por partido para compartir en redes.
export default async function Image({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const m = await getMatch(decodeURIComponent(id));
  const pc = (x: number) => `${Math.round(x * 100)}%`;

  const home = m?.equipo_home ?? "Partido";
  const away = m?.equipo_away ?? "";
  const ph = m ? m.prob_home : 0.34;
  const pd = m ? m.prob_draw : 0.33;
  const pa = m ? m.prob_away : 0.33;
  const top = m?.top_scorelines?.[0];

  return new ImageResponse(
    (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          width: "100%",
          height: "100%",
          backgroundColor: "#0a0b0e",
          color: "#e7e9ee",
          padding: "64px",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 30 }}>
          <span style={{ color: "#5b9cf6", fontWeight: 700 }}>mundialista·26</span>
          <span style={{ color: "#9aa1ad" }}>Mundial 2026</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", alignItems: "center", fontSize: 58, fontWeight: 700 }}>
            <span>{home}</span>
            <span style={{ color: "#686f7b", margin: "0 20px" }}>vs</span>
            <span>{away}</span>
          </div>

          <div
            style={{
              display: "flex",
              width: "100%",
              height: 28,
              marginTop: 40,
              borderRadius: 14,
              overflow: "hidden",
              backgroundColor: "#181b22",
            }}
          >
            <div style={{ display: "flex", width: `${ph * 100}%`, backgroundColor: "#5b9cf6" }} />
            <div style={{ display: "flex", width: `${pd * 100}%`, backgroundColor: "#767f8d" }} />
            <div style={{ display: "flex", width: `${pa * 100}%`, backgroundColor: "#a986e8" }} />
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 18, fontSize: 32 }}>
            <span>{home} {pc(ph)}</span>
            <span style={{ color: "#9aa1ad" }}>Empate {pc(pd)}</span>
            <span>{away} {pc(pa)}</span>
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 27, color: "#9aa1ad" }}>
          <span>{top ? `Marcador probable: ${top.score} (${pc(top.prob)})` : "Predicción del modelo"}</span>
          <span>Elo + Dixon-Coles + ensemble</span>
        </div>
      </div>
    ),
    { ...size }
  );
}
