import React from 'react';

export default class ColorWheelCreator extends React.Component {
    constructor(props) {
        super(props);
        this.state = {colors : [{"value": "#000000", "key": "0"}]}
        this.addColor = this.addColor.bind(this);
        this.onColorChange = this.onColorChange.bind(this)
    }

    render() {
        return(
            <div>
                <button type="button" onClick={this.addColor}>add</button>
                <button type="button">remove</button>
                <input type="number"/>
                {this.state.colors.map((c) => {
                    return(
                        <div>
                            <ColorBox value={c.value} key={c.key} onColorChange={this.onColorChange} />
                        </div>
                    );
                })}
            </div>
        );
    }

    addColor() {
        this.setState((prevState, props) => {
            let newColor = {"value": "#000000", "key": prevState.colors.length.toString()};
            return({colors : [prevState.colors, newColor]});
        });
    }

    onColorChange(key, value) {
        let newColors = this.state.colors;
        newColors[parseInt(key, 10)][value] = value;
        this.setState({colors : newColors});
    }
}

function ColorBox(props) {
    function onChange(e) {
        props.onColorChange(props.key, e.value);
    }
    console.log(props.key) //TODO: Why is key uindefined??
    return(
        <input type="color" value={props.value} onChange={onChange} />
    );
}
