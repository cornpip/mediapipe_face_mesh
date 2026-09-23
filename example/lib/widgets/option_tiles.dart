import 'package:flutter/material.dart';

const TextStyle _labelStyle = TextStyle(
  fontSize: 13,
  fontWeight: FontWeight.w600,
  color: Colors.black87,
);

BoxDecoration _tileDecoration() => BoxDecoration(
  color: Colors.black.withValues(alpha: 0.035),
  border: Border.all(color: Colors.black12),
  borderRadius: BorderRadius.circular(12),
);

/// A collapsible group of option tiles.
class OptionsPanel extends StatelessWidget {
  const OptionsPanel({
    super.key,
    required this.title,
    required this.children,
    this.initiallyExpanded = false,
  });

  final String title;
  final List<Widget> children;
  final bool initiallyExpanded;

  @override
  Widget build(BuildContext context) {
    final RoundedRectangleBorder shape = RoundedRectangleBorder(
      side: const BorderSide(color: Colors.black12),
      borderRadius: BorderRadius.circular(12),
    );
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 0),
      child: ExpansionTile(
        initiallyExpanded: initiallyExpanded,
        shape: shape,
        collapsedShape: shape,
        backgroundColor: Colors.black.withValues(alpha: 0.035),
        collapsedBackgroundColor: Colors.black.withValues(alpha: 0.035),
        tilePadding: const EdgeInsets.symmetric(horizontal: 14),
        // Top room for the floating label of a first dropdown.
        childrenPadding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
        title: Text(title, style: _labelStyle),
        children: children,
      ),
    );
  }
}

/// A labeled switch. A null [onChanged] disables it.
class SwitchTile extends StatelessWidget {
  const SwitchTile({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final bool value;
  final ValueChanged<bool>? onChanged;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: _tileDecoration(),
      padding: const EdgeInsets.only(left: 14),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: _labelStyle,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          Transform.scale(
            scale: 0.78,
            child: Switch(
              value: value,
              onChanged: onChanged,
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
          ),
        ],
      ),
    );
  }
}

/// A label, the current angle, and a rotate-by-90° button. A null
/// [onRotate] disables the button.
class RotationTile extends StatelessWidget {
  const RotationTile({
    super.key,
    required this.label,
    required this.tooltip,
    required this.degrees,
    required this.onRotate,
  });

  final String label;
  final String tooltip;
  final int degrees;
  final VoidCallback? onRotate;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: _tileDecoration(),
      padding: const EdgeInsets.only(left: 14, right: 4),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: _labelStyle,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          Text(
            '$degrees°',
            style: const TextStyle(fontSize: 13, color: Colors.black87),
          ),
          IconButton(
            icon: const Icon(Icons.rotate_right, size: 20),
            tooltip: tooltip,
            onPressed: onRotate,
          ),
        ],
      ),
    );
  }
}

/// A dropdown styled like the option tiles. A null [onChanged] disables it.
class LabeledDropdown<T> extends StatelessWidget {
  const LabeledDropdown({
    super.key,
    required this.label,
    required this.value,
    required this.items,
    required this.onChanged,
    this.isExpanded = false,
  });

  final String label;
  final T? value;
  final List<DropdownMenuItem<T>> items;
  final ValueChanged<T?>? onChanged;
  final bool isExpanded;

  @override
  Widget build(BuildContext context) {
    OutlineInputBorder border(Color color, [double width = 1]) =>
        OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: color, width: width),
        );
    return DropdownButtonFormField<T>(
      // initialValue needs Flutter 3.35; the package supports 3.32.
      // ignore: deprecated_member_use
      value: value,
      isDense: true,
      isExpanded: isExpanded,
      borderRadius: BorderRadius.circular(12),
      style: _labelStyle,
      icon: const Icon(Icons.expand_more_rounded, size: 20),
      decoration: InputDecoration(
        labelText: label,
        isDense: true,
        filled: true,
        fillColor: Colors.black.withValues(alpha: 0.035),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 14,
          vertical: 12,
        ),
        labelStyle: const TextStyle(fontSize: 12, color: Colors.black54),
        floatingLabelStyle: const TextStyle(fontSize: 12.5),
        border: border(Colors.black12),
        enabledBorder: border(Colors.black12),
        focusedBorder: border(Colors.black38, 1.4),
      ),
      items: items,
      onChanged: onChanged,
    );
  }
}
