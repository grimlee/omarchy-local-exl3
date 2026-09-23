import QtQuick
import QtQuick.Controls as Controls
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Ui

Panel {
  id: root
  moduleName: "grimlee.local-exl3"
  ipcTarget: "grimlee.local-exl3"
  manageIpc: false

  readonly property string sourceDir: String(Qt.resolvedUrl("..")).replace(/^file:\/\//, "").replace(/\/$/, "")
  readonly property string cli: sourceDir + "/bin/local-exl3"
  readonly property string logPath: (Quickshell.env("XDG_STATE_HOME") || (Quickshell.env("HOME") + "/.local/state")) + "/omarchy-local-exl3/server.log"
  readonly property color ink: bar ? bar.foreground : Color.foreground
  readonly property color dim: Qt.darker(ink, 1.6)
  readonly property color bg: Color.popups.background
  readonly property string mono: bar ? bar.fontFamily : Style.font.family
  property var snap: ({ running: false, health: false, model_found: false, runtime_found: false,
                       stable_mm: false, stable_mm_implementation: false, profile: "balanced",
                       gpu_name: null, vram_used_mib: null, vram_total_mib: null, endpoint: "http://127.0.0.1:8881/v1" })
  property string selectedProfile: "balanced"
  property bool receivedStatus: false
  property string message: ""
  property bool editingPath: false
  readonly property bool busy: action.running
  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  function refresh() { if (!poll.running) poll.running = true }
  function run(args) {
    if (action.running) return
    message = "Working…"
    action.command = [cli].concat(args)
    action.running = true
  }
  function accept(json) {
    try {
      var value = JSON.parse(json)
      if (!value || typeof value.running !== "boolean") return
      var wasRunning = snap.running
      snap = value
      if (!receivedStatus || wasRunning || value.running) selectedProfile = value.profile
      receivedStatus = true
    } catch (e) { message = "Could not read status" }
  }
  onOpenedChanged: if (opened) refresh()

  Process {
    id: poll
    command: [root.cli, "status", "--json"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.accept(text) }
  }
  Process {
    id: action
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.message = text.trim() }
    stderr: StdioCollector { waitForEnd: true; onStreamFinished: if (text.trim()) root.message = text.trim().split("\n").slice(-2).join(" · ") }
    onExited: function(code) { if (code !== 0 && root.message === "Working…") root.message = "Action failed · see Logs"; root.refresh() }
  }
  Process { id: logs; command: ["omarchy-launch-tui", "--app-id=org.omarchy.local-exl3-log", "less", "+G", root.logPath] }
  Process { id: openPi; command: ["omarchy-launch-tui", "--app-id=org.omarchy.local-exl3-pi", root.cli, "open-pi"] }
  Timer { interval: root.opened ? 4000 : 20000; running: true; repeat: true; triggeredOnStart: true; onTriggered: root.refresh() }

  BarIconButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    iconComponent: Component {
      Text { text: "◆"; color: root.snap.health ? root.ink : root.dim; font.pixelSize: Style.font.bodySmall }
    }
    tooltipText: "Local EXL3 · " + (root.snap.health ? "Ready" : root.snap.running ? "Starting" : "Stopped")
    onPressed: root.toggle()
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    padding: 0
    contentWidth: fittedContentWidth(Style.space(365))
    contentHeight: fittedContentHeight(body.implicitHeight)
    Rectangle { anchors.fill: parent; color: root.bg }
    Column {
      id: body
      anchors.left: parent.left; anchors.right: parent.right
      anchors.margins: Style.space(16)
      spacing: Style.space(10)
      Text { text: "LOCAL EXL3"; color: root.ink; font.family: root.mono; font.pixelSize: 14 }
      Text { width: parent.width; text: root.snap.model || "Qwen3.8-27B EXL3"; color: root.ink; font.family: root.mono; wrapMode: Text.WordWrap }
      Text {
        width: parent.width
        text: "GPU  " + (root.snap.gpu_name || "Unavailable") + "  ·  " +
              (root.snap.vram_used_mib === null ? "—" : (root.snap.vram_used_mib / 1024).toFixed(1)) +
              " / " + (root.snap.vram_total_mib === null ? "—" : (root.snap.vram_total_mib / 1024).toFixed(1)) + " GiB"
        color: root.dim; font.family: root.mono; wrapMode: Text.WordWrap
      }
      Text { text: "PROFILE"; color: root.dim; font.family: root.mono }
      Button {
        text: (root.selectedProfile === "balanced" ? "◉ " : "○ ") + "Balanced · Q4 KV · 117,760 ctx · Recommended"
        width: parent.width; leftAlign: true; bordered: true; foreground: root.ink; fontFamily: root.mono
        enabled: !root.busy && !root.snap.running
        onClicked: root.selectedProfile = "balanced"
      }
      Button {
        text: (root.selectedProfile === "max-context" ? "◉ " : "○ ") + "Max Context · Q3 KV · 150,016 ctx"
        width: parent.width; leftAlign: true; bordered: true; foreground: root.ink; fontFamily: root.mono
        enabled: !root.busy && !root.snap.running
        onClicked: root.selectedProfile = "max-context"
      }
      Text { text: "Stable MM  " + (root.snap.stable_mm ? "Enabled" : root.snap.stable_mm_implementation ? "Ready for launch" : "Implementation missing"); color: root.snap.stable_mm_implementation ? root.ink : Color.urgent; font.family: root.mono }
      Text { text: "Status  " + (root.snap.health ? "Ready" : root.snap.running ? "Starting / unhealthy" : "Stopped"); color: root.ink; font.family: root.mono }
      Text { text: "Endpoint  " + root.snap.endpoint; color: root.dim; font.family: root.mono; font.pixelSize: Style.font.bodySmall }
      Text { text: "Model  " + (root.snap.model_found ? "Found" : "Not found"); color: root.snap.model_found ? root.ink : Color.urgent; font.family: root.mono }
      Row {
        spacing: Style.space(6)
        Button { text: "Start"; enabled: !root.busy && !root.snap.running && root.snap.model_found && root.snap.runtime_found && root.snap.stable_mm_implementation; bordered: true; foreground: root.ink; fontFamily: root.mono; onClicked: root.run(["start", "--profile", root.selectedProfile]) }
        Button { text: "Stop"; enabled: !root.busy && root.snap.running; bordered: true; foreground: root.ink; fontFamily: root.mono; onClicked: root.run(["stop"]) }
        Button { text: "Restart"; enabled: !root.busy && root.snap.running; bordered: true; foreground: root.ink; fontFamily: root.mono; onClicked: root.run(["restart", "--profile", root.snap.profile]) }
      }
      Row {
        spacing: Style.space(6)
        Button { text: "Logs"; bordered: true; foreground: root.ink; fontFamily: root.mono; onClicked: logs.running = true }
        Button { text: "Open Pi"; enabled: root.snap.health; bordered: true; foreground: root.ink; fontFamily: root.mono; onClicked: openPi.running = true }
        Button { text: "Configure Model Path"; bordered: true; foreground: root.ink; fontFamily: root.mono; onClicked: root.editingPath = !root.editingPath }
      }
      Controls.TextField {
        id: modelPath
        visible: root.editingPath
        width: parent.width
        placeholderText: "/absolute/path/to/model"
        color: root.ink; font.family: root.mono
        background: Rectangle { color: root.bg; border.color: root.dim }
        onAccepted: root.run(["configure-model", text])
      }
      Button { visible: root.editingPath; text: "Save path"; enabled: modelPath.text.length > 0; bordered: true; foreground: root.ink; fontFamily: root.mono; onClicked: root.run(["configure-model", modelPath.text]) }
      Text { visible: root.message !== ""; width: parent.width; text: root.message; color: root.dim; font.family: root.mono; wrapMode: Text.WordWrap }
    }
  }
}
