import { execSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const repoRoot = path.resolve(process.cwd(), '..', '..')
const publicDir = path.resolve(process.cwd(), 'public')
const outputPath = path.join(publicDir, 'build-info.json')

const now = new Date().toISOString()
let commit = null
let branch = null

try {
  commit = execSync(`git -C "${repoRoot}" rev-parse HEAD`, { encoding: 'utf8' }).trim()
} catch {
  commit = null
}

try {
  branch = execSync(`git -C "${repoRoot}" rev-parse --abbrev-ref HEAD`, { encoding: 'utf8' }).trim()
} catch {
  branch = null
}

const payload = {
  built_at: now,
  commit,
  branch,
}

fs.mkdirSync(publicDir, { recursive: true })
fs.writeFileSync(outputPath, JSON.stringify(payload, null, 2))
console.log(`Wrote ${path.relative(repoRoot, outputPath)}`)
