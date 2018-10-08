import React from 'react';
import ColorWheelCreator from './ColorWheelCreator.js'
import model from '../data-model.json';

export default class AnimationCreator extends React.Component {
    constructor(props) {
        super(props);
        let defaultAnimTypeName = "switch"
        this.state = {
            animType: defaultAnimTypeName,
            animSubtype: Object.keys(model.animTypes[defaultAnimTypeName]["subtypes"])[0]
        }
        this.onTypeChange = this.onTypeChange.bind(this);
    }

    render() {
        let globalFields = this.generateFields(model.globalFields);
        let relativeFields = this.generateFields(model.animTypes[this.state.animType]["subtypes"][this.state.animSubtype]["fields"])
        return(
            <div>
                {globalFields}
                {relativeFields}
            </div>
        );
    }

    generateFields(fields) {
        /*This function assumes there is no more than a single animTypeSelect field
        and a single animSubtype field. Any other field types may occur
        in any number
        */
        return(
            fields.map((o) => {
                if (o.type === "number") {
                    return (
                      <div>
                          <label>{o.label}</label><input key={o.name} type="number"/>
                      </div>
                    );
                } else if (o.type === "text") {
                    return (
                      <div>
                          <label>{o.label}</label><input key={o.name} type="text"/>
                      </div>
                    );
                } else if (o.type === "select") {
                  return (
                      <div>
                          <label>{o.label}</label>
                          <select>
                              {o.values.map((v) => {
                                  return <option value={v}>{v}</option>
                              })}
                          </select>
                      </div>
                    );
                } else if (o.type === "animTypeSelect") {
                    let types = model.animTypes;
                    let typeNames = Object.keys(types);
                    return(
                        <div>
                            <label>{o.label}</label>
                            <select value={this.state.type} onChange={this.onTypeChange}>
                                {typeNames.map((t) => {
                                    return <option key={types[t].name} value={types[t].name}>{types[t].label}</option>
                                })}
                            </select>
                        </div>
                    );
                } else if (o.type === "animSubtypeSelect") {
                  let subtypes = model.animTypes[this.state.animType]["subtypes"];
                  let subtypeNames = Object.keys(subtypes);
                    return(
                        <div>
                            <label>{o.label}</label>
                            <select>
                                {subtypeNames.map((st) => {
                                    return <option key={subtypes[st].name} value={subtypes[st].name}>{subtypes[st].label}</option>
                                })}
                            </select>
                        </div>
                    );
                }
                else if (o.type === "colorWheel") {
                    return <ColorWheelCreator />
                } else {
                    return <h3>Invalid Field</h3>
                }
            })
        );
    }

    onTypeChange(e) {
        this.setState({
            animType: e.target.value,
            animSubtype: Object.keys(model.animTypes[e.target.value]["subtypes"])[0]
        })
    }
}
