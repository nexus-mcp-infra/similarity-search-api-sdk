#!/usr/bin/env node
/**
 * Entry point — nexus-similarity-search-api
 *
 * Transport is selected via env var TRANSPORT:
 *   stdio (default) — for local dev, MCP Inspector, subprocess integration
 *   http             — Streamable HTTP, stateless, for remote/prod deployment
 */

import "dotenv/config";
import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { SERVER_NAME, SERVER_VERSION, CORE_BASE_URL } from "./constants.js";
import { registerTools } from "./tools.js";

// Fail fast on misconfiguration instead of failing on first tool call.
const EnvSchema = z.object({
  NEXUS_CORE_BASE_URL: z.string().url().optional(),
  NEXUS_CORE_API_KEY: z.string().optional(),
  TRANSPORT: z.enum(["stdio", "http"]).optional(),
  PORT: z.string().optional(),
});
EnvSchema.parse(process.env);

function buildServer(): McpServer {
  const server = new McpServer({ name: SERVER_NAME, version: SERVER_VERSION });
  registerTools(server);
  return server;
}

async function runStdio(): Promise<void> {
  const server = buildServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`${SERVER_NAME} v${SERVER_VERSION} — stdio ready`);
}

async function runHttp(): Promise<void> {
  const app = express();
  app.use(express.json({ limit: "10mb" }));

  app.get("/health", (_req, res) => {
    res.json({ status: "healthy", server: SERVER_NAME, version: SERVER_VERSION, core: CORE_BASE_URL });
  });

  app.post("/mcp", async (req, res) => {
    // Stateless: a fresh server + transport per request avoids request-id
    // collisions across concurrent clients and keeps this horizontally
    // scalable without sticky sessions.
    const server = buildServer();
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });
    res.on("close", () => transport.close());
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  });

  const port = parseInt(process.env.PORT ?? "3000", 10);
  const httpServer = app.listen(port, () => {
    console.error(`${SERVER_NAME} v${SERVER_VERSION} — http ready on :${port}/mcp`);
  });

  for (const signal of ["SIGTERM", "SIGINT"] as const) {
    process.on(signal, () => {
      console.error(`${signal} received, shutting down gracefully`);
      httpServer.close(() => process.exit(0));
    });
  }
}

const transportMode = process.env.TRANSPORT ?? "stdio";
const run = transportMode === "http" ? runHttp : runStdio;

run().catch((err) => {
  console.error("Fatal error starting server:", err);
  process.exit(1);
});
