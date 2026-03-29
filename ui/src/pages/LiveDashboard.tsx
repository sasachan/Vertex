import { useVertexStream } from "../hooks/useVertexStream.ts";
import { TopBar } from "../components/layout/TopBar.tsx";
import { VideoFeed } from "../components/live/VideoFeed.tsx";
import { StatePanel } from "../components/live/StatePanel.tsx";
import { BioGauges } from "../components/live/BioGauges.tsx";
import { ShotTable } from "../components/live/ShotTable.tsx";
import { StatsBar } from "../components/live/StatsBar.tsx";
import { Card } from "../components/shared/StatusDot.tsx";

export function LiveDashboard() {
  const stream = useVertexStream();

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <TopBar fps={stream.fps} status={stream.status} />

      {/* Main content — responsive: column on mobile, row on desktop */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Video feed — takes most space */}
        <div className="flex-1 p-3 min-w-0">
          <VideoFeed />
        </div>

        {/* Right panel — state + bio + shots */}
        <div className="w-full lg:w-80 flex flex-col gap-3 p-3 pt-0 lg:pt-3 overflow-y-auto border-l border-vertex-border">
          <Card title="State">
            <StatePanel state={stream.state} hold={stream.hold} />
          </Card>

          <Card title="Biomechanics">
            <BioGauges bio={stream.bio} />
          </Card>

          <Card title="Shot History">
            <ShotTable shots={stream.shots} />
          </Card>
        </div>
      </div>

      {/* Bottom stats bar */}
      <StatsBar
        shotCount={stream.shotCount}
        avgHold={stream.avgHold}
        bestHold={stream.bestHold}
        vertexScore={stream.vertexScore}
      />
    </div>
  );
}
