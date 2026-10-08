"use client";

import React, { useState } from "react";
import { BrandHero } from "@/components/BrandHero";
import { LoanPredictorStudio } from "@/components/LoanPredictorStudio";
import { AnimatePresence, motion } from "motion/react";

export default function App() {
  const [activeView, setActiveView] = useState<"hero" | "predictor">("hero");
  const [isDark, setIsDark] = useState<boolean>(true);

  // Toggle dark class on documentElement
  const toggleTheme = () => {
    const nextDark = !isDark;
    setIsDark(nextDark);
    if (nextDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      <AnimatePresence mode="wait">
        {activeView === "hero" ? (
          <motion.div
            key="hero-view"
            initial={{ opacity: 0, scale: 0.99 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.99 }}
            transition={{ duration: 0.3 }}
          >
            <BrandHero
              onLaunchPredictor={() => setActiveView("predictor")}
              isDark={isDark}
              onToggleTheme={toggleTheme}
            />
          </motion.div>
        ) : (
          <motion.div
            key="predictor-view"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <LoanPredictorStudio
              onBackToHero={() => setActiveView("hero")}
              isDark={isDark}
              onToggleTheme={toggleTheme}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
