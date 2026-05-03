import { Switch, Route, Router as WouterRouter, Link, useLocation } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Waves, GitCompare, Bookmark } from "lucide-react";
import NotFound from "@/pages/not-found";
import { DiscoverPage } from "@/pages/Discover";
import { ComparePage } from "@/pages/Compare";
import { SavedPage } from "@/pages/Saved";
import { setBaseUrl } from "@workspace/api-client-react";
import { cn } from "@/lib/utils";

setBaseUrl("");

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, refetchOnWindowFocus: false } },
});

function NavLink({ href, icon: Icon, children }: { href: string; icon: typeof Waves; children: React.ReactNode }) {
  const [loc] = useLocation();
  const active = loc === href || (href !== "/" && loc.startsWith(href));
  return (
    <Link
      href={href}
      className={cn(
        "inline-flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors",
        active ? "bg-primary text-primary-foreground" : "hover:bg-muted text-foreground/80",
      )}
      data-testid={`nav-${href.replace("/", "") || "home"}`}
    >
      <Icon className="h-4 w-4" />
      {children}
    </Link>
  );
}

function Shell() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/40">
      <header className="border-b bg-card/60 backdrop-blur sticky top-0 z-10">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-500 grid place-items-center">
              <Waves className="h-4 w-4 text-white" />
            </div>
            <div>
              <div className="font-bold leading-tight">Cross-Lingual Homophone Explorer</div>
              <div className="text-xs text-muted-foreground leading-tight">
                Real TTS · MFCC · Dynamic Time Warping
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-1">
            <NavLink href="/" icon={Waves}>Discover</NavLink>
            <NavLink href="/compare" icon={GitCompare}>Compare</NavLink>
            <NavLink href="/saved" icon={Bookmark}>Saved</NavLink>
          </nav>
        </div>
      </header>
      <main className="container mx-auto px-4 py-6 max-w-6xl">
        <Switch>
          <Route path="/" component={DiscoverPage} />
          <Route path="/compare" component={ComparePage} />
          <Route path="/saved" component={SavedPage} />
          <Route component={NotFound} />
        </Switch>
      </main>
      <footer className="container mx-auto px-4 py-6 text-center text-xs text-muted-foreground">
        Acoustic similarity computed from synthesized PCM audio · 13-coefficient MFCC + Δ + ΔΔ ·
        Cosine-distance DTW
      </footer>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
          <Shell />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
