import assert from "node:assert/strict";
import { test } from "node:test";
// @ts-expect-error -- explicit .ts extension is required by Node's type stripping
import { hoursSince, isCheckDue, rewriteApiUrl } from "../lib/timeline.ts";

const now = Date.parse("2026-10-26T12:00:00Z");

test("UTC timestamps are read as UTC, not local time", () => {
  assert.equal(hoursSince("2026-10-26T06:30:00Z", now), 5.5);
});

test("check due exactly at 24 h, not before", () => {
  assert.equal(isCheckDue(["2026-10-25T12:00:01Z"], now), false);
  assert.equal(isCheckDue(["2026-10-25T12:00:00Z"], now), true);
});

test("only the latest photo decides", () => {
  assert.equal(isCheckDue(["2026-10-20T00:00:00Z", "2026-10-26T11:00:00Z"], now), false);
});

test("no photos means no reminder", () => {
  assert.equal(isCheckDue([], now), false);
});

test("localhost API is rewritten for a phone on the LAN", () => {
  assert.equal(rewriteApiUrl("http://localhost:8001", "192.168.1.20"), "http://192.168.1.20:8001");
  assert.equal(rewriteApiUrl("http://localhost:8001", "localhost"), "http://localhost:8001");
  assert.equal(rewriteApiUrl("http://api.example.org", "192.168.1.20"), "http://api.example.org");
  assert.equal(rewriteApiUrl("http://localhost:8001", null), "http://localhost:8001");
});
