"use client";

import React from "react";
import {
  ShieldCheck,
  Cpu,
  ArrowRight,
  Sparkles,
  Moon,
  Sun,
  Scale,
} from "lucide-react";
import { OriginButton } from "@/components/ui/origin-button";
import { SonarGrid } from "@/components/ui/sonar-grid";
import { VerticalCutReveal } from "@/components/ui/vertical-cut-reveal";
import { Button } from "@/components/ui/button";
import { motion, useReducedMotion } from "motion/react";

interface BrandHeroProps {
  onLaunchPredictor: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

export function BrandHero({ onLaunchPredictor, isDark, onToggleTheme }: BrandHeroProps) {
  const reduce = useReducedMotion();

  const enter = (delay: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 14, filter: "blur(6px)" },
          animate: { opacity: 1, y: 0, filter: "blur(0px)" },
          transition: { duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] as const },
        };

  return (
    <SonarGrid
      id="creditlens-sonar-hero"
      spacing={26}
      dotRadius={1.2}
      baseOpacity={isDark ? 0.24 : 0.18}
      pingEvery={3.8}
      speed={160}
      ringWidth={100}
      amplitude={2.0}
      interactive={true}
      pingArea={[0.2, 0.2, 0.8, 0.8]}
      className="relative h-screen w-full flex flex-col justify-between overflow-hidden bg-background text-foreground select-none"
    >
      {/* Soft radial background wash to maintain crisp text contrast while sonar rings pass underneath */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_50%_45%_at_50%_50%,var(--color-background)_0%,transparent_100%)] opacity-90"
      />

      {/* Floating Island Navbar */}
      <header className="relative z-20 w-full pt-4 sm:pt-5 px-4 sm:px-6">
        <div className="max-w-4xl mx-auto flex items-center justify-between rounded-2xl border border-border/80 bg-background/80 dark:bg-card/70 backdrop-blur-xl px-4 py-2.5 shadow-lg shadow-black/5 dark:shadow-black/20">
          {/* Brand Logo */}
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-500 to-sky-600 flex items-center justify-center shadow-md shadow-emerald-500/20 text-white font-bold">
              <ShieldCheck className="w-4.5 h-4.5" />
            </div>
            <span className="font-extrabold text-base sm:text-lg tracking-tight text-foreground">
              CreditLens
            </span>
          </div>

          {/* Right Controls */}
          <div className="flex items-center space-x-3">
            <div className="hidden sm:flex items-center space-x-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5 font-medium">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                LightGBM v1.0
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Scale className="w-3.5 h-3.5 text-muted-foreground" />
                RBI-Aligned Design
              </span>
            </div>

            <div className="h-4 w-px bg-border/60 hidden sm:block" />

            {/* Dark/Light Toggle */}
            <Button
              variant="outline"
              size="icon"
              onClick={onToggleTheme}
              className="h-8 w-8 rounded-xl border-border/80 bg-card/60 backdrop-blur-sm"
              title="Toggle theme"
            >
              {isDark ? (
                <Sun className="h-4 w-4 text-amber-400" />
              ) : (
                <Moon className="h-4 w-4 text-sky-600" />
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Center Content (Zero-Scroll, Above the Fold) */}
      <main className="relative z-10 flex-1 flex flex-col justify-center items-center px-4 sm:px-6 max-w-4xl mx-auto text-center">
        {/* Compliance & Portfolio Pill */}
        <motion.p
          {...enter(0)}
          className="text-muted-foreground border-border/80 bg-muted/40 backdrop-blur-sm mb-4 inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-medium shadow-sm"
        >
          <Sparkles className="w-3.5 h-3.5 text-emerald-500" />
          <span className="font-semibold text-foreground">Portfolio project</span>
          <span>·</span>
          <span>Aligned with RBI Digital Lending Guidelines</span>
        </motion.p>

        {/* Project Name with Vertical Cut Reveal Character Animation */}
        <div className="flex flex-col items-center justify-center gap-2 mb-3">
          <h1 className="font-black text-6xl sm:text-7xl md:text-8xl text-foreground tracking-tighter flex items-center justify-center">
            <VerticalCutReveal
              splitBy="characters"
              staggerDuration={0.04}
              staggerFrom="center"
              transition={{ damping: 20, stiffness: 300, type: "spring" }}
              containerClassName="justify-center items-center"
            >
              CreditLens
            </VerticalCutReveal>
          </h1>
          <motion.p
            {...enter(0.08)}
            className="text-xs sm:text-sm font-semibold tracking-wider uppercase text-emerald-500 font-mono"
          >
            Explainable Credit Underwriting
          </motion.p>
        </div>

        {/* Hero Headline */}
        <motion.h2
          {...enter(0.12)}
          className="text-2xl sm:text-3xl md:text-4xl font-extrabold tracking-tight text-foreground leading-[1.2] max-w-2xl"
        >
          See why every loan is approved or declined.
        </motion.h2>

        {/* Subtitle */}
        <motion.p
          {...enter(0.18)}
          className="mt-4 text-xs sm:text-sm md:text-base text-muted-foreground max-w-2xl font-normal leading-relaxed"
        >
          A calibrated LightGBM model with a cost-weighted approval threshold and SHAP-based reasons for every decision.
        </motion.p>

        {/* Centerpiece: Interactive OriginButton (Full Ripple Coverage) */}
        <motion.div
          {...enter(0.24)}
          className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4 w-full sm:w-auto"
        >
          <OriginButton
            onClick={onLaunchPredictor}
            className="w-full sm:w-auto h-14 px-9 rounded-2xl shadow-xl shadow-primary/20 text-base font-semibold group cursor-pointer"
          >
            <span className="flex items-center gap-2.5">
              <Cpu className="w-5 h-5 text-emerald-500 transition-transform duration-200 group-hover:rotate-12" />
              <span>Launch Loan Predictor</span>
              <ArrowRight className="w-4 h-4 ml-1 transition-transform duration-200 group-hover:translate-x-1" />
            </span>
          </OriginButton>
        </motion.div>

        {/* Helper Line */}
        <motion.p
          {...enter(0.3)}
          className="mt-5 text-[11px] sm:text-xs text-muted-foreground/80 tracking-wide font-mono"
        >
          No sign-up needed
        </motion.p>
      </main>

      {/* Minimal Bottom Bar */}
      <footer className="relative z-10 w-full border-t border-border/40 bg-background/60 backdrop-blur-sm py-3 px-6 text-center text-xs text-muted-foreground flex flex-col sm:flex-row items-center justify-between gap-2 max-w-7xl mx-auto">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-foreground">CreditLens Underwriting Platform</span>
          <span>•</span>
          <span>Zero Data Leakage Pipeline</span>
        </div>
        <div className="flex items-center space-x-3 text-[11px] font-mono text-muted-foreground">
          <span>LightGBM • TreeSHAP • Indian Rupee (₹) • React</span>
        </div>
      </footer>
    </SonarGrid>
  );
}
