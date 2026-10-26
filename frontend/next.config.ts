import { networkInterfaces } from "node:os";
import type { NextConfig } from "next";

// Let a phone on the same Wi-Fi open the dev server at http://<this-machine's-IP>:3000.
// Next blocks cross-origin dev requests by default, so this machine's own LAN
// IPv4 addresses are allowed explicitly (read at startup; restart after
// changing networks).
const lanAddresses = Object.values(networkInterfaces())
  .flat()
  .filter((i) => i && i.family === "IPv4" && !i.internal)
  .map((i) => i!.address);

const nextConfig: NextConfig = {
  allowedDevOrigins: lanAddresses,
};

export default nextConfig;
