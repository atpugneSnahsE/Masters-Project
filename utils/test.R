# ==============================================================================
# ResNet34 Justification: Comparative Analysis of CARLA and VIL100
# Publication-Quality Figures | Scientific Visualization
# ==============================================================================

rm(list = ls())

# ==============================================================================
# 1. PACKAGES
# ==============================================================================

required_packages <- c(
  "tidyverse",
  "ggplot2",
  "ggrepel",       # Non-overlapping labels on scatter plots
  "patchwork",     # Compositing multiple panels
  "scales",        # Axis formatting
  "RColorBrewer"   # Colorblind-safe palettes
)

new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]
if (length(new_packages) > 0) install.packages(new_packages)

library(tidyverse)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(scales)
library(RColorBrewer)

# ==============================================================================
# 2. DESIGN SYSTEM
# ==============================================================================

# Colorblind-safe palette (Wong 2011, Nature Methods)
COL_CARLA    <- "#0072B2"   # Blue
COL_VIL100   <- "#E69F00"   # Amber
COL_RESNET34 <- "#D55E00"   # Vermilion — accent for highlight
COL_MUTED    <- "#CCCCCC"   # All other models
COL_GRID     <- "#EBEBEB"

THEME_BASE <- theme_classic(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle    = element_text(size = 11, color = "grey40", hjust = 0),
    axis.title       = element_text(size = 12),
    axis.text        = element_text(size = 10, color = "grey20"),
    legend.position  = "top",
    legend.title     = element_blank(),
    legend.text      = element_text(size = 10),
    panel.grid.major = element_line(color = COL_GRID, linewidth = 0.4),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text       = element_text(face = "bold", size = 11),
    plot.margin      = margin(12, 16, 12, 12)
  )

# Helper: mark ResNet34 rows
flag_resnet <- function(df) {
  df %>% mutate(
    is_resnet = grepl("(?i)resnet.?34", Model),
    pt_color  = if_else(is_resnet, COL_RESNET34, NA_character_),
    pt_size   = if_else(is_resnet, 5, 3),
    pt_alpha  = if_else(is_resnet, 1, 0.65),
    pt_shape  = if_else(is_resnet, 18, 16)   # diamond vs circle
  )
}

# ==============================================================================
# 3. LOAD & PREPARE DATA
# ==============================================================================

carla_path  <- "/Users/mac/Downloads/carla_comparison_20260609_1739.csv"
vil100_path <- "/Users/mac/Downloads/vil100_comparison_20260609_1921.csv"

carla  <- read.csv(carla_path,  stringsAsFactors = FALSE, check.names = FALSE)
vil100 <- read.csv(vil100_path, stringsAsFactors = FALSE, check.names = FALSE)

colnames(carla)  <- trimws(colnames(carla))
colnames(vil100) <- trimws(colnames(vil100))

carla$Dataset  <- "CARLA"
vil100$Dataset <- "VIL100"

combined <- bind_rows(carla, vil100)

numeric_cols <- c("mIoU", "Precision", "Recall", "FPS",
                  "Params_M", "Train_Time_min", "Epochs")
for (col in numeric_cols) {
  if (col %in% colnames(combined))
    combined[[col]] <- as.numeric(as.character(combined[[col]]))
}

combined <- flag_resnet(combined)

# Order models by mean mIoU descending (for sorted dot plots)
model_order <- combined %>%
  group_by(Model) %>%
  summarise(mean_mIoU = mean(mIoU, na.rm = TRUE)) %>%
  arrange(desc(mean_mIoU)) %>%
  pull(Model)

combined$Model <- factor(combined$Model, levels = rev(model_order))

cat("✓ Data loaded:", nrow(combined), "rows\n")
print(combined %>% select(Model, Dataset, mIoU, Precision, Recall, FPS,
                          Params_M, Train_Time_min) %>% head(8))

# ==============================================================================
# FIGURE 1: Cleveland Dot Plot — mIoU Comparison
# (Scientific standard for ranked multi-group comparisons)
# ==============================================================================

fig1 <- ggplot(
  combined,
  aes(x = mIoU, y = Model)
) +
  # Connecting lines between CARLA and VIL100 per model
  geom_line(
    aes(group = Model),
    color = "grey70",
    linewidth = 0.6,
    linetype = "dashed"
  ) +
  # All model points
  geom_point(
    aes(
      color  = Dataset,
      shape  = Dataset,
      size   = is_resnet,
      alpha  = is_resnet
    )
  ) +
  # Highlight ResNet34 model name
  geom_text(
    data = combined %>% filter(is_resnet, Dataset == "VIL100"),
    aes(label = "◄ ResNet34"),
    hjust = -0.15,
    size  = 3.5,
    color = COL_RESNET34,
    fontface = "bold"
  ) +
  scale_color_manual(values = c("CARLA" = COL_CARLA, "VIL100" = COL_VIL100)) +
  scale_shape_manual(values = c("CARLA" = 16, "VIL100" = 17)) +
  scale_size_manual(values = c("TRUE" = 4.5, "FALSE" = 3), guide = "none") +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.7), guide = "none") +
  THEME_BASE +
  labs(
    title    = "Lane Detection mIoU Across Backbone Architectures",
    subtitle = "CARLA (simulation) vs VIL100 (real-world) | Dashed lines connect same model across datasets",
    x        = "Mean Intersection over Union (mIoU)",
    y        = NULL
  )

# ==============================================================================
# FIGURE 2: Efficiency Frontier — mIoU vs FPS
# (Core justification: accuracy-speed trade-off; ResNet34 as Pareto-optimal)
# ==============================================================================

fig2 <- ggplot(
  combined,
  aes(x = FPS, y = mIoU, color = Dataset, shape = Dataset)
) +
  # Soft background reference quadrant: high FPS, high mIoU
  annotate(
    "rect",
    xmin = median(combined$FPS, na.rm = TRUE),
    xmax = Inf,
    ymin = median(combined$mIoU, na.rm = TRUE),
    ymax = Inf,
    fill  = "#E8F4F8",
    alpha = 0.5
  ) +
  annotate(
    "text",
    x     = median(combined$FPS, na.rm = TRUE) + 1,
    y     = max(combined$mIoU, na.rm = TRUE) - 0.01,
    label = "Optimal zone\n(high accuracy + high speed)",
    hjust = 0,
    size  = 3.2,
    color = "grey50",
    fontface = "italic"
  ) +
  geom_point(
    aes(
      size  = is_resnet,
      alpha = is_resnet
    )
  ) +
  geom_label_repel(
    aes(label = Model),
    size         = 3.2,
    box.padding  = 0.4,
    point.padding = 0.3,
    min.segment.length = 0.2,
    show.legend  = FALSE,
    fontface     = if_else(
      combined$is_resnet, "bold", "plain"
    ),
    color        = if_else(
      combined$is_resnet, COL_RESNET34, "grey30"
    )
  ) +
  scale_color_manual(values = c("CARLA" = COL_CARLA, "VIL100" = COL_VIL100)) +
  scale_shape_manual(values = c("CARLA" = 16, "VIL100" = 17)) +
  scale_size_manual(values = c("TRUE" = 5.5, "FALSE" = 3), guide = "none") +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.65), guide = "none") +
  THEME_BASE +
  labs(
    title    = "Accuracy–Speed Trade-off: mIoU vs Frames Per Second",
    subtitle = "Upper-right quadrant = optimal zone | Label weight indicates ResNet34",
    x        = "Inference Speed (FPS)",
    y        = "mIoU"
  )

# ==============================================================================
# FIGURE 3: mIoU vs Parameters (Bubble = FPS)
# (Model complexity justification)
# ==============================================================================

fig3 <- ggplot(
  combined,
  aes(x = Params_M, y = mIoU)
) +
  geom_point(
    aes(
      size  = FPS,
      color = Dataset,
      shape = Dataset,
      alpha = is_resnet
    )
  ) +
  geom_label_repel(
    aes(
      label    = Model,
      color    = Dataset,
      fontface = if_else(is_resnet, "bold", "plain")
    ),
    size          = 3.2,
    box.padding   = 0.45,
    show.legend   = FALSE
  ) +
  scale_color_manual(values = c("CARLA" = COL_CARLA, "VIL100" = COL_VIL100)) +
  scale_shape_manual(values = c("CARLA" = 16, "VIL100" = 17)) +
  scale_size_continuous(name = "FPS", range = c(3, 10)) +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.6), guide = "none") +
  THEME_BASE +
  guides(
    color = guide_legend(override.aes = list(size = 4)),
    size  = guide_legend(title = "FPS")
  ) +
  labs(
    title    = "Accuracy vs Model Complexity",
    subtitle = "Bubble size = inference speed (FPS) | Bolder label = ResNet34",
    x        = "Parameters (Millions)",
    y        = "mIoU"
  )

# ==============================================================================
# FIGURE 4: Normalized Performance Heatmap
# (Shows ResNet34's balanced multi-metric profile — strongest holistic argument)
# ==============================================================================

normalize_01 <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (diff(rng) == 0) return(rep(0.5, length(x)))
  (x - rng[1]) / diff(rng)
}

# For Params_M and Train_Time_min: lower is better, so invert
heatmap_data <- combined %>%
  group_by(Model) %>%
  summarise(
    mIoU          = mean(mIoU,          na.rm = TRUE),
    Precision     = mean(Precision,     na.rm = TRUE),
    Recall        = mean(Recall,        na.rm = TRUE),
    FPS           = mean(FPS,           na.rm = TRUE),
    Params_M      = mean(Params_M,      na.rm = TRUE),
    Train_Time    = mean(Train_Time_min, na.rm = TRUE)
  ) %>%
  mutate(
    norm_mIoU      = normalize_01(mIoU),
    norm_Precision = normalize_01(Precision),
    norm_Recall    = normalize_01(Recall),
    norm_FPS       = normalize_01(FPS),
    norm_Params    = 1 - normalize_01(Params_M),    # Invert: fewer params = better
    norm_TrainTime = 1 - normalize_01(Train_Time)   # Invert: less time = better
  ) %>%
  pivot_longer(
    cols      = starts_with("norm_"),
    names_to  = "Metric",
    values_to = "Score"
  ) %>%
  mutate(
    Metric = recode(Metric,
                    norm_mIoU      = "mIoU",
                    norm_Precision = "Precision",
                    norm_Recall    = "Recall",
                    norm_FPS       = "Speed (FPS)",
                    norm_Params    = "Efficiency\n(Params↓)",
                    norm_TrainTime = "Train\nEfficiency"
    ),
    is_resnet = grepl("(?i)resnet.?34", Model)
  )

# Sort model axis: ResNet34 on top, rest by mean score
model_score_order <- heatmap_data %>%
  group_by(Model) %>%
  summarise(mean_score = mean(Score)) %>%
  arrange(desc(mean_score)) %>%
  pull(Model)

heatmap_data$Model <- factor(heatmap_data$Model, levels = rev(model_score_order))

resnet34_label_data <- heatmap_data %>%
  filter(is_resnet) %>%
  group_by(Model) %>%
  summarise(xpos = n_distinct(Metric) / 2 + 0.5, ypos = as.numeric(Model[1]))

fig4 <- ggplot(
  heatmap_data,
  aes(x = Metric, y = Model, fill = Score)
) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(
    aes(label = sprintf("%.2f", Score)),
    size  = 3.2,
    color = if_else(heatmap_data$Score > 0.55, "white", "grey20")
  ) +
  # Bold border around ResNet34 row
  geom_tile(
    data    = heatmap_data %>% filter(is_resnet),
    color   = COL_RESNET34,
    fill    = NA,
    linewidth = 1.2
  ) +
  scale_fill_gradientn(
    colors = c("#F7F7F7", "#92C5DE", "#2166AC"),
    values = c(0, 0.5, 1),
    name   = "Normalized\nScore",
    limits = c(0, 1),
    breaks = c(0, 0.5, 1),
    labels = c("Low", "Mid", "High")
  ) +
  THEME_BASE +
  theme(
    axis.text.y    = element_text(
      face  = if_else(levels(heatmap_data$Model) %in%
                        (heatmap_data %>% filter(is_resnet) %>% pull(Model) %>% unique()),
                      "bold", "plain"),
      color = if_else(levels(heatmap_data$Model) %in%
                        (heatmap_data %>% filter(is_resnet) %>% pull(Model) %>% unique()),
                      COL_RESNET34, "grey20")
    ),
    panel.grid.major = element_blank(),
    legend.position  = "right"
  ) +
  labs(
    title    = "Normalized Multi-Metric Performance Profile",
    subtitle = "Higher = better across all metrics | Red border = ResNet34 | Params & Train Time inverted (lower is better)",
    x        = NULL,
    y        = NULL
  )

# ==============================================================================
# FIGURE 5: Lollipop — Precision & Recall by Model per Dataset
# ==============================================================================

pr_long <- combined %>%
  pivot_longer(
    cols      = c(Precision, Recall),
    names_to  = "Metric",
    values_to = "Value"
  )

fig5 <- ggplot(
  pr_long,
  aes(x = Value, y = Model, color = Dataset)
) +
  geom_segment(
    aes(x = 0, xend = Value, y = Model, yend = Model, alpha = is_resnet),
    linewidth = 0.9
  ) +
  geom_point(
    aes(
      shape = Dataset,
      size  = is_resnet,
      alpha = is_resnet
    )
  ) +
  facet_wrap(~Metric, ncol = 2) +
  scale_color_manual(values = c("CARLA" = COL_CARLA, "VIL100" = COL_VIL100)) +
  scale_shape_manual(values = c("CARLA" = 16, "VIL100" = 17)) +
  scale_size_manual(values = c("TRUE" = 4.5, "FALSE" = 2.8), guide = "none") +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.6), guide = "none") +
  THEME_BASE +
  labs(
    title    = "Precision & Recall Across Architectures",
    subtitle = "Opaque lollipops = ResNet34",
    x        = "Score",
    y        = NULL
  )

# ==============================================================================
# FIGURE 6: Training Efficiency — Time vs Epochs (Lollipop)
# ==============================================================================

fig6_data <- combined %>%
  filter(!is.na(Train_Time_min)) %>%
  mutate(time_per_epoch = Train_Time_min / Epochs)

fig6 <- ggplot(
  fig6_data,
  aes(x = Train_Time_min, y = Model)
) +
  geom_segment(
    aes(
      x     = 0,
      xend  = Train_Time_min,
      y     = Model,
      yend  = Model,
      color = Dataset,
      alpha = is_resnet
    ),
    linewidth = 1
  ) +
  geom_point(
    aes(
      color = Dataset,
      shape = Dataset,
      size  = is_resnet,
      alpha = is_resnet
    )
  ) +
  geom_text(
    data    = fig6_data %>% filter(is_resnet),
    aes(
      label = paste0(Train_Time_min, " min"),
      x     = Train_Time_min
    ),
    hjust    = -0.2,
    size     = 3.2,
    color    = COL_RESNET34,
    fontface = "bold"
  ) +
  facet_wrap(~Dataset, ncol = 2) +
  scale_color_manual(values = c("CARLA" = COL_CARLA, "VIL100" = COL_VIL100)) +
  scale_shape_manual(values = c("CARLA" = 16, "VIL100" = 17)) +
  scale_size_manual(values = c("TRUE" = 4.5, "FALSE" = 2.8), guide = "none") +
  scale_alpha_manual(values = c("TRUE" = 1, "FALSE" = 0.55), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  THEME_BASE +
  labs(
    title    = "Training Time Across Backbones",
    subtitle = "Highlighted = ResNet34 | Lower is computationally preferred",
    x        = "Total Training Time (Minutes)",
    y        = NULL
  )

# ==============================================================================
# COMPOSITE PANEL: Figures 1 + 2 (Summary justification panel)
# ==============================================================================

composite_panel <- (fig1 | fig2) +
  plot_annotation(
    title   = "ResNet34 Backbone: Accuracy and Efficiency Justification",
    caption = "Data: CARLA (simulation) & VIL100 (real-world) benchmarks",
    theme   = theme(
      plot.title   = element_text(face = "bold", size = 16, hjust = 0),
      plot.caption = element_text(color = "grey50", size = 9)
    )
  )

# ==============================================================================
# 4. SAVE ALL FIGURES
# ==============================================================================

save_fig <- function(plot_obj, filename, w = 10, h = 6.5) {
  ggsave(
    filename = filename,
    plot     = plot_obj,
    width    = w,
    height   = h,
    dpi      = 600,
    bg       = "white"
  )
  cat("✓ Saved:", filename, "\n")
}

save_fig(fig1,            "Fig1_mIoU_Cleveland_DotPlot.png",       w = 9,  h = 6)
save_fig(fig2,            "Fig2_Efficiency_Frontier.png",           w = 10, h = 6.5)
save_fig(fig3,            "Fig3_Accuracy_Complexity_Bubble.png",    w = 10, h = 6.5)
save_fig(fig4,            "Fig4_Normalized_Heatmap.png",            w = 11, h = 5.5)
save_fig(fig5,            "Fig5_Precision_Recall_Lollipop.png",     w = 11, h = 6)
save_fig(fig6,            "Fig6_TrainingTime_Lollipop.png",         w = 11, h = 5.5)
save_fig(composite_panel, "Fig0_Summary_Panel.png",                 w = 18, h = 7)

# ==============================================================================
# 5. SUMMARY STATISTICS
# ==============================================================================

summary_stats <- combined %>%
  group_by(Dataset, is_resnet) %>%
  summarise(
    Avg_mIoU      = mean(mIoU,          na.rm = TRUE),
    Avg_Precision = mean(Precision,     na.rm = TRUE),
    Avg_Recall    = mean(Recall,        na.rm = TRUE),
    Avg_FPS       = mean(FPS,           na.rm = TRUE),
    Avg_Params_M  = mean(Params_M,      na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(Group = if_else(is_resnet, "ResNet34", "Other Models"))

write.csv(summary_stats, "Summary_Statistics.csv", row.names = FALSE)

cat("\n==============================================\n")
cat(" ResNet34 vs Other Models Summary\n")
cat("==============================================\n")
print(summary_stats %>% select(Dataset, Group, Avg_mIoU, Avg_Precision, Avg_FPS))
cat("\n✓ All figures and summary saved.\n")
cat("==============================================\n")