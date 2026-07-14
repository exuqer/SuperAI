import { spawn } from 'node:child_process'
import net from 'node:net'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const webDir = path.dirname(fileURLToPath(import.meta.url))
const appDir = path.resolve(webDir, '..')
const rootDir = path.resolve(appDir, '..')
const isWindows = process.platform === 'win32'
const children = []

function start(command, args, cwd) {
  const child = spawn(command, args, { cwd, shell: isWindows, stdio: 'inherit' })
  children.push(child)
  child.on('exit', code => {
    if (code && code !== 0) process.exitCode = code
  })
  return child
}

function isListening(port) {
  return new Promise(resolve => {
    const socket = net.createConnection({ host: '127.0.0.1', port })
    socket.once('connect', () => { socket.destroy(); resolve(true) })
    socket.once('error', () => resolve(false))
  })
}

if (!await isListening(8000)) {
  start('python', ['-m', 'uvicorn', 'server.server:app', '--host', '127.0.0.1', '--port', '8000'], rootDir)
}
start(isWindows ? 'npm.cmd' : 'npm', ['run', 'dev:frontend'], appDir)

function stop() {
  for (const child of children) child.kill()
}

process.on('SIGINT', stop)
process.on('SIGTERM', stop)
